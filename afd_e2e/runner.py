"""Owned subprocess lifecycle, evaluation and cleanup."""

from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, evaluation
from .config import TOOL_ROOT, Case
from .metrics import capture_metrics
from .performance import points
from .planner import Service, build_services, environment, network, runtime_environment
from .reporting import (
    LOG_FAILURE, ROOT_EXCEPTION, is_profiler_advisory, prepare_output_directory,
    primary_error, report_text, write_json,
)
from .routing import routes
from .process_utils import (
    close_process_identities,
    find_processes_matching_environment,
    kill_processes_matching_environment,
    terminate_process_groups,
)


def exit_code(status: str) -> int:
    if status == "passed":
        return 0
    if status == "cancelled":
        return 130
    return 1


class ManagedProcess:
    def __init__(
        self, name: str, command: tuple[str, ...], env: dict[str, str], run_dir: Path
    ):
        self.name = name
        self.reader_error: Exception | None = None
        self.phase = "inference"
        self.diagnostics = {"inference": [], "shutdown": []}
        self.root_errors = {"inference": None, "shutdown": None}
        self.advisories = []
        self.process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self.reader = threading.Thread(
            target=self._read, args=(run_dir / f"{name}.log",), daemon=True
        )
        self.reader.start()

    def _read(self, path: Path) -> None:
        try:
            with path.open("w", encoding="utf-8") as log:
                for number, line in enumerate(self.process.stdout, start=1):
                    log.write(line)
                    log.flush()
                    if LOG_FAILURE.search(line):
                        phase = self.phase
                        advisory = is_profiler_advisory(line)
                        evidence = (
                            self.advisories
                            if advisory
                            else self.diagnostics[phase]
                        )
                        # Repeated wrapper errors must not hide the later root cause.
                        if len(evidence) == 20:
                            del evidence[10]
                        row = {
                            "log": path.name,
                            "line": number,
                            "message": line.strip(),
                            **({"phase": phase} if advisory else {}),
                        }
                        evidence.append(row)
                        if (
                            not advisory and self.root_errors[phase] is None
                            and ROOT_EXCEPTION.search(line)
                        ):
                            self.root_errors[phase] = row
                    print(f"[{self.name}] {line}", end="", flush=True)
        except Exception as exc:
            self.reader_error = exc
        finally:
            self.process.stdout.close()

    def ensure_alive(self) -> None:
        if self.reader_error:
            raise RuntimeError(f"{self.name} log writer failed: {self.reader_error}")
        if self.diagnostics["inference"]:
            evidence = self.root_errors["inference"] or self.diagnostics["inference"][0]
            raise RuntimeError(
                f"{evidence['log']}:{evidence['line']}: {evidence['message']}"
            )
        status = self.process.poll()
        if status is not None:
            raise RuntimeError(f"{self.name} exited unexpectedly with status {status}")

    def wait(
        self, timeout: float, services: list[ManagedProcess], progress=None
    ) -> None:
        deadline = time.monotonic() + timeout
        while self.process.poll() is None:
            if progress:
                progress()
            for service in services:
                service.ensure_alive()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{self.name} exceeded {timeout}s")
            time.sleep(0.2)
        self.reader.join(timeout=5)
        if self.reader.is_alive() or self.reader_error:
            raise RuntimeError(
                f"{self.name} log writer did not complete: {self.reader_error}"
            )
        if self.process.returncode:
            raise RuntimeError(
                f"{self.name} exited with status {self.process.returncode}; see {self.name}.log"
            )


def wait_ready(case: Case, services: list[ManagedProcess], progress=None) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    pending = {route["url"] + "/v1/models" for route in routes(case.to_dict())}
    deadline = time.monotonic() + case.startup_timeout
    last_error = None
    expected = "afd-e2e-baseline" if case.baseline else "afd-e2e-attention"
    while time.monotonic() < deadline:
        if progress:
            progress()
        for service in services:
            service.ensure_alive()
        for url in sorted(pending):
            try:
                with opener.open(url, timeout=3) as response:
                    data = json.load(response)
                    if response.status == 200 and any(
                        model["id"] == expected for model in data["data"]
                    ):
                        print(f"API ready: {url}", flush=True)
                        pending.remove(url)
            except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
                last_error = exc
        if not pending:
            return
        time.sleep(0.5)
    raise TimeoutError(f"API startup timed out at {sorted(pending)}; last error={last_error!r}")


def source_manifest(case: Case) -> dict:
    source = {
        str(path.relative_to(TOOL_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (TOOL_ROOT / "afd_e2e").rglob("*.py")
    }
    git = None
    if not case.baseline and (case.repo / ".git").exists():
        git = {}
        for key, arguments in (
            ("revision", ("rev-parse", "HEAD")),
            ("status", ("status", "--short")),
            ("diff", ("diff", "HEAD")),
        ):
            try:
                result = subprocess.run(
                    ("git", "-C", str(case.repo), *arguments),
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                git[key] = result.stdout if result.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                git[key] = None
    return {"harness_version": __version__, "source_sha256": source,
            "afd_plugin": git}


def _preflight_command(case: Case, devices: tuple[str, ...]) -> tuple[str, ...]:
    _, nic = network(case)
    ports = (
        tuple(case.api_port + i for i in range(2 * len(case.baseline_layouts)))
        if case.baseline else (case.api_port, case.api_port + 1, case.afd_port)
    )
    probe_cmd = [
        case.python,
        "-m",
        "afd_e2e.probe",
        "--model",
        case.model_path,
        "--devices",
        ",".join(devices),
        "--api-host",
        case.api_host,
        "--ports",
        ",".join(map(str, ports)),
    ]
    if nic:
        probe_cmd.extend(("--nic", nic))
    return tuple(probe_cmd)


def _stop_children(
    case: Case,
    run_id: str,
    children: list[ManagedProcess],
    services: list[ManagedProcess],
    exit_codes: dict[str, int | None],
) -> list[str]:
    failures = []
    try:
        failures.extend(
            terminate_process_groups(
                [child.process for child in children],
                termination_timeout_s=case.cleanup_timeout,
                poll_interval_s=0.2,
                reap_timeout_s=5,
            )
        )
        if case.comm == "async" and services:
            failures.extend(
                kill_processes_matching_environment(
                    {"AFD_E2E_RUN_ID": run_id, "AFD_E2E_PROCESS_ROLE": "ffn"},
                    timeout_s=120,
                    poll_interval_s=0.2,
                    process_name="FFN",
                )
            )
        for child in children:
            child.reader.join(timeout=5)
            exit_codes[child.name] = child.process.poll()
            if child.reader.is_alive() or child.reader_error:
                failures.append(
                    f"{child.name} log reader incomplete: {child.reader_error}"
                )
    except BaseException as exc:
        failures.append(f"Process cleanup error: {exc}")
    return failures


def execute(case: Case) -> dict:
    if case.devices is None:
        raise ValueError(
            "Specify --device/--devices or inherit ASCEND_RT_VISIBLE_DEVICES "
            "through npu run"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = case.run_dir or TOOL_ROOT / "runs" / run_id
    prepare_output_directory(run_dir)
    config_path = run_dir / "resolved-config.json"
    # Claim a plan-only directory before another run can reuse it.
    config_path.touch(exist_ok=False)
    write_json(config_path, case.to_dict())
    summary = {
        "run_id": run_id,
        "case_id": case.case_id,
        "run_dir": str(run_dir),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stage": "devices",
        "errors": [],
        "process_exit_codes": {},
        "inference_status": "running",
        "shutdown_status": "not_started",
        "cleanup_status": "not_started",
        "stage_durations": {},
        "test_results": {kind: "not_started" for kind in case.tests},
    }
    started = stage_started = time.monotonic()
    last_progress = 0.0
    reporting_errors = []

    def save_json(path, value):
        # Reporting failures must not interrupt process cleanup.
        try:
            write_json(path, value)
        except OSError as exc:
            error = f"Report write failed ({path.name}): {exc}"
            if error not in reporting_errors:
                reporting_errors.append(error)
                summary["errors"].append(error)
            summary["status"] = "failed"
            summary.setdefault("failed_stage", "reporting")

    def progress(force=False):
        nonlocal last_progress
        now = time.monotonic()
        if force or now - last_progress >= 15:
            summary["updated_at"] = datetime.now(timezone.utc).isoformat()
            summary["elapsed_seconds"] = round(now - started, 3)
            summary["stage_elapsed_seconds"] = round(now - stage_started, 3)
            snapshot = summary.copy()
            if not summary.get("finished_at"):
                snapshot["status"] = "running"
            save_json(run_dir / "summary.json", snapshot)
            try:
                print(
                    f"[{summary['stage']}] elapsed={summary['elapsed_seconds']:.1f}s "
                    f"stage={summary['stage_elapsed_seconds']:.1f}s output={run_dir}",
                    flush=True,
                )
            except OSError as exc:
                error = f"Progress output failed: {exc}"
                if error not in reporting_errors:
                    reporting_errors.append(error)
                    summary["errors"].append(error)
                summary["status"] = "failed"
                summary.setdefault("failed_stage", "reporting")
            last_progress = now

    def stage(name):
        nonlocal stage_started
        now = time.monotonic()
        summary["stage_durations"][summary["stage"]] = round(now - stage_started, 3)
        stage_started = now
        summary["stage"] = name
        summary["stage_started_at"] = datetime.now(timezone.utc).isoformat()
        progress(force=True)

    summary["stage_started_at"] = summary["started_at"]
    progress(force=True)
    children: list[ManagedProcess] = []
    services: list[ManagedProcess] = []
    history = []
    current_test = None
    manifest = source_manifest(case)
    write_json(run_dir / "manifest.json", manifest)
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    def cancel(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    def start_job(
        name: str, command: tuple[str, ...], env: dict[str, str], timeout: float
    ) -> None:
        history.append({"name": name, "command": command})
        write_json(run_dir / "commands.json", history)
        child = ManagedProcess(name, command, env, run_dir)
        children.append(child)
        child.wait(timeout, services, progress)

    def start_workload(name: str, kind: str) -> None:
        start_job(
            name,
            (
                case.python,
                "-m",
                "afd_e2e.workloads",
                "--config",
                str(config_path),
                "--kind",
                kind,
                "--output",
                str(run_dir / f"{name}.json"),
            ),
            job_env,
            case.eval_timeout,
        )

    for sig in previous_handlers:
        signal.signal(sig, cancel)
    try:
        manifest["devices"] = case.devices
        manifest["unused_devices"] = case.devices[case.required_devices :]
        write_json(run_dir / "manifest.json", manifest)
        specifications = build_services(case, case.devices, run_id)
        if "perf" in case.tests:
            manifest["performance_points"] = points(case.to_dict()["performance"])
            summary["warnings"] = list(dict.fromkeys(
                warning for point in manifest["performance_points"]
                for warning in point["warnings"]
            ))
            for warning in summary["warnings"]:
                print(f"Warning: {warning}", flush=True)
        manifest["services"] = [service.to_dict() for service in specifications]
        manifest["routes"] = routes(case.to_dict())
        write_json(run_dir / "manifest.json", manifest)
        probe_service = Service(
            "probe",
            case.devices,
            (),
            environment(case, "probe", case.devices, run_id),
        )
        job_env = runtime_environment(probe_service)
        stage("preflight")
        start_job(
            "preflight",
            _preflight_command(case, case.devices),
            job_env,
            case.startup_timeout,
        )
        probe_lines = (run_dir / "preflight.log").read_text().splitlines()
        reports = [
            json.loads(line.split("AFD_E2E_PROBE ", 1)[1])
            for line in probe_lines
            if "AFD_E2E_PROBE " in line
        ]
        if len(reports) != 1:
            raise RuntimeError("Preflight did not produce a readiness report")
        manifest["readiness"] = reports[0]
        write_json(run_dir / "manifest.json", manifest)
        stage("startup")
        for service in specifications:
            print(
                f"Starting {service.role}: devices={','.join(service.devices)}",
                flush=True,
            )
            history.append(service.to_dict())
            write_json(run_dir / "commands.json", history)
            child = ManagedProcess(
                service.role, service.command, runtime_environment(service), run_dir
            )
            children.append(child)
            services.append(child)
            child.ensure_alive()
        wait_ready(case, services, progress)
        summary["metrics"] = {
            "before": capture_metrics(case.to_dict(), run_dir, "before")
        }
        workload_started = time.time()
        summary["workload_started"] = workload_started
        for kind in case.tests:
            current_test = kind
            summary["test_results"][kind] = "running"
            if kind == "smoke":
                stage("smoke")
                start_workload("smoke", "smoke")
            elif kind == "perf":
                stage("performance")
                start_job(
                    "performance",
                    (
                        case.python,
                        "-m",
                        "afd_e2e.performance",
                        "--config",
                        str(config_path),
                    ),
                    job_env,
                    case.eval_timeout,
                )
            else:
                if case.dataset_path:
                    stage("dataset")
                    start_job(
                        "dataset",
                        (
                            case.python, "-m", "afd_e2e.evaluation",
                            "--dataset", str(case.dataset_path),
                            "--output", str(run_dir / "gsm8k-task"),
                        ),
                        job_env, case.startup_timeout,
                    )
                stage("gsm8k")
                start_job(
                    "evaluation", evaluation.command(case, run_dir),
                    job_env, case.eval_timeout,
                )
                try:
                    summary["gsm8k"] = evaluation.read_results(
                        run_dir / "gsm8k", case.limit, case.threshold
                    )
                except evaluation.AccuracyError as exc:
                    # A score below threshold does not make the service unusable.
                    summary["gsm8k"] = exc.result
                    summary["test_results"][kind] = "failed"
                    summary["status"] = "failed"
                    summary.setdefault("failed_stage", "gsm8k")
                    summary["errors"].append(f"AccuracyError: {exc}")
            if summary["test_results"][kind] == "running":
                summary["test_results"][kind] = "passed"
            for service in services:
                service.ensure_alive()
            current_test = None
        if summary["status"] == "running":
            summary["status"] = "passed"
    except BaseException as exc:
        summary["status"] = (
            "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        summary.setdefault("failed_stage", summary["stage"])
        if current_test is not None:
            summary["test_results"][current_test] = summary["status"]
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        # Complete bounded cleanup even if cancellation is repeated.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if "metrics" in summary:
            try:
                summary["metrics"]["after"] = capture_metrics(
                    case.to_dict(), run_dir, "after"
                )
            except Exception as exc:
                summary["status"] = "failed"
                summary.setdefault("failed_stage", "reporting")
                summary["errors"].append(f"Metrics capture failed: {exc}")
        summary["shutdown_status"] = "running"
        summary["cleanup_status"] = "running"
        summary["inference_status"] = summary["status"]
        inference_stage = summary["stage"]
        for child in children:
            child.phase = "shutdown"
        stage("cleanup")
        cleanup_failures = _stop_children(
            case, run_id, children, services, summary["process_exit_codes"]
        )
        # A cancelled performance job saves its partial measurements while stopping.
        if "perf" in case.tests:
            try:
                metrics = run_dir / "performance.json"
                if metrics.is_file():
                    summary["performance"] = json.loads(metrics.read_text())
            except (OSError, ValueError) as exc:
                summary["status"] = summary["inference_status"] = "failed"
                summary["failed_stage"] = "performance"
                summary["test_results"]["perf"] = "failed"
                summary["errors"].append(f"Performance result read failed: {exc}")
        forced = [
            error for error in cleanup_failures if error.startswith("forced SIGKILL")
        ]
        cleanup_failures = [error for error in cleanup_failures if error not in forced]
        diagnostics = {}
        for service in services:
            phases = {phase: list(rows) for phase, rows in service.diagnostics.items()}
            for phase, root in service.root_errors.items():
                if root and root not in phases[phase]:
                    phases[phase].append(root)
                    phases[phase].sort(key=lambda row: row["line"])
            diagnostics[service.name] = phases
        summary["log_diagnostics"] = diagnostics
        summary["log_advisories"] = {
            service.name: service.advisories for service in services
        }
        shutdown_errors = list(forced)
        for phases in diagnostics.values():
            for phase, evidence in phases.items():
                if not evidence:
                    continue
                first = primary_error(evidence)
                error = f"{phase}: {first['log']}:{first['line']}: {first['message']}"
                if phase == "shutdown":
                    shutdown_errors.append(error)
                else:
                    if summary["inference_status"] == "passed":
                        summary["inference_status"] = "failed"
                        summary["status"] = "failed"
                        summary.setdefault("failed_stage", inference_stage)
                    if not any(first["message"] in item for item in summary["errors"]):
                        summary["errors"].append(error)
        summary["shutdown_status"] = "failed" if shutdown_errors else "passed"
        if shutdown_errors:
            summary["status"] = "failed"
            summary.setdefault("failed_stage", "shutdown")
            summary["errors"].extend(shutdown_errors)
        if cleanup_failures:
            summary["status"] = "failed"
            summary.setdefault("failed_stage", "cleanup")
            summary["errors"].extend(cleanup_failures)
            summary["cleanup_failed"] = True
        summary["cleanup_status"] = "failed" if cleanup_failures else "passed"
        try:
            survivors = find_processes_matching_environment(
                {"AFD_E2E_RUN_ID": run_id}
            )
            try:
                if survivors:
                    raise RuntimeError(
                        f"Owned processes still alive: {[p.pid for p in survivors]}"
                    )
            finally:
                close_process_identities(survivors)
        except Exception as exc:
            summary["status"] = "failed"
            summary.setdefault("failed_stage", "cleanup")
            summary["cleanup_failed"] = True
            summary["cleanup_status"] = "failed"
            summary["errors"].append(f"Process cleanup: {exc}")
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        save_json(run_dir / "diagnostics.json", diagnostics)
        save_json(run_dir / "advisories.json", summary["log_advisories"])
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        stage("finished")
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if reporting_errors:
            summary["status"] = "failed"
            summary.setdefault("failed_stage", "reporting")
        save_json(run_dir / "summary.json", summary)
        try:
            (run_dir / "report.txt").write_text(report_text(summary), encoding="utf-8")
        except OSError as exc:
            summary["status"] = "failed"
            summary.setdefault("failed_stage", "reporting")
            summary["errors"].append(f"Report write failed (report.txt): {exc}")
        try:
            (run_dir / "exit-code.txt").write_text(f"{exit_code(summary['status'])}\n")
        except OSError as exc:
            summary["status"] = "failed"
            summary.setdefault("failed_stage", "reporting")
            summary["errors"].append(f"Report write failed (exit-code.txt): {exc}")
        save_json(run_dir / "summary.json", summary)
    print(report_text(summary), end="", flush=True)
    return summary
