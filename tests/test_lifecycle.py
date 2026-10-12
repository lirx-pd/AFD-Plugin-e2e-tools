import json
import os
import signal
import socket
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from afd_e2e import planner, runner
from afd_e2e.reporting import PROFILER_PARSE_ADVISORY
from afd_e2e.process_utils import (
    kill_processes_matching_environment,
    terminate_process_groups,
)


@pytest.fixture
def simulated_runtime(monkeypatch, tmp_path):
    fixture = Path(__file__).with_name("fake_runtime.py")
    native_process = runner.ManagedProcess

    def process(name, command, env, run_dir):
        if name in {"preflight", "evaluation"}:
            command = (sys.executable, str(fixture), name, *command)
        return native_process(name, command, env, run_dir)

    def services(case, devices, run_id):
        return tuple(
            replace(s, command=(sys.executable, str(fixture), "service", *s.command))
            for s in planner.build_services(case, devices, run_id)
        )

    monkeypatch.setattr(runner, "ManagedProcess", process)
    monkeypatch.setattr(runner, "build_services", services)
    monkeypatch.setenv("AFD_E2E_FAKE_ACTIVITY", str(tmp_path / "activity"))
    # Keep both API ports reserved while choosing a free consecutive pair.
    while True:
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            try:
                second.bind(("127.0.0.1", port + 1))
                break
            except OSError:
                pass
    repo = tmp_path / "afd-plugin"
    (repo / "afd_plugin").mkdir(parents=True)
    return [
        "--repo",
        str(repo),
        "--api-port",
        str(port),
        "--afd-port",
        "6455",
        "--startup-timeout",
        "10",
        "--eval-timeout",
        "10",
        "--cleanup-timeout",
        "2",
        "--output",
        str(tmp_path / "result"),
        "--limit",
        "7",
    ]


@pytest.mark.parametrize(
    "flags",
    [
        ["--dbo", "on", "--execution", "graph"],
        ["--comm", "async", "--mrv", "mrv2"],
        ["--model", "v4-flash", "--model-path", "/weights", "--mrv", "mrv2"],
        [
            "--model",
            "v4-flash",
            "--model-path",
            "/weights",
            "--comm",
            "async",
            "--mrv",
            "mrv2",
            "--layered-gmm",
            "on",
        ],
    ],
)
def test_run_lifecycle_with_real_http_and_children(
    make_case, simulated_runtime, flags
):
    devices = ",".join(map(str, range(16 if "v4-flash" in flags else 4)))
    case = make_case(*simulated_runtime, "--device", devices, *flags)
    summary = runner.execute(case)
    assert summary["status"] == "passed", summary["errors"]
    assert (
        summary["inference_status"]
        == summary["shutdown_status"]
        == summary["cleanup_status"]
        == "passed"
    )
    assert summary["stage"] == "finished"
    assert {"startup", "smoke", "gsm8k", "cleanup"} <= summary["stage_durations"].keys()
    assert summary["elapsed_seconds"] > 0
    assert "Result: passed" in (case.run_dir / "report.txt").read_text()
    assert summary["gsm8k"]["samples"] == 7
    assert all(code is not None for code in summary["process_exit_codes"].values())
    manifest = json.loads((case.run_dir / "manifest.json").read_text())
    assert manifest["readiness"]["simulated"]
    assert manifest["devices"] == list(case.devices)
    assert manifest["unused_devices"] == []
    assert (case.run_dir / "exit-code.txt").read_text() == "0\n"
    if case.dbo:
        assert (
            len(
                json.loads((case.run_dir / "dbo-workload.json").read_text())["requests"]
            )
            == 24
        )


def test_accuracy_failure_still_cleans_up_children(
    make_case, simulated_runtime, monkeypatch
):
    monkeypatch.setenv("AFD_E2E_FAKE_SCORE", "0.1")
    summary = runner.execute(make_case(*simulated_runtime, "--device", "0,1,2,3"))
    assert summary["status"] == "failed"
    assert any("strict-match" in error for error in summary["errors"])
    assert summary["gsm8k"]["strict_match"] == 0.1
    assert summary["gsm8k"]["threshold"] == 0.27
    assert summary["failed_stage"] == "gsm8k"
    assert "GSM8K: 10.00%" in (Path(summary["run_dir"]) / "report.txt").read_text()
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_native_multi_instance_performance_lifecycle(make_case, simulated_runtime, tmp_path):
    case = make_case(
        *simulated_runtime, "--baseline", "--topology", "dp1,dp2",
        "--repo", tmp_path / "missing-plugin", "--device", "0,1,2",
        "--perf-mode", "matrix", "--perf-input-lens", "8",
        "--perf-output-lens", "4", "--perf-concurrencies", "3",
        "--perf-requests", "6", "--perf-warmup", "3",
    )
    summary = runner.execute(case)
    assert summary["status"] == "passed", summary["errors"]
    assert summary["cleanup_status"] == "passed"
    assert summary["coverage"]["status"] == "not_applicable"
    assert "gsm8k" not in summary
    assert {"baseline-0", "baseline-1"} <= summary["process_exit_codes"].keys()
    manifest = json.loads((case.run_dir / "manifest.json").read_text())
    assert len(manifest["services"]) == 2 and len(manifest["routes"]) == 3
    records = [json.loads(line) for line in (
        case.run_dir / "performance/requests.jsonl"
    ).read_text().splitlines()]
    measured = [r for r in records if r["phase"] == "measured"]
    assert [r["route"]["domain"] for r in measured] == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("comm", ["sync", "async"])
def test_v4_performance_lifecycle_reuses_devices_and_keeps_warmup_separate(
    make_case, simulated_runtime, comm
):
    case = make_case(
        *simulated_runtime,
        "--device",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "--model",
        "v4-flash",
        "--model-path",
        "/weights",
        "--comm",
        comm,
        "--test",
        "perf",
        "--perf-input-lens",
        "8,32",
        "--perf-output-lens",
        "4",
        "--perf-concurrencies",
        "1,3",
        "--perf-requests",
        "4",
        "--perf-warmup",
        "1",
    )
    summary = runner.execute(case)
    assert summary["status"] == "passed", summary["errors"]
    assert "gsm8k" not in summary
    assert len(summary["performance"]["points"]) == 6
    for point in summary["performance"]["points"]:
        assert (
            point["completed"]
            == point["metrics"]["ttft_ms"]["count"]
            == point["metrics"]["tpot_ms"]["count"]
            == 4
        )
        assert point["warmup"]["completed"] == 1
        assert point["peak_concurrency"] <= point["concurrency"]
        assert point["achieved_rps"] > 0 and point["output_tokens_per_second"] > 0
        assert sum(group["completed"] for group in point["by_length"]) == 4
    rows = [
        json.loads(line)
        for line in (case.run_dir / "performance" / "requests.jsonl")
        .read_text()
        .splitlines()
    ]
    assert sum(row["phase"] == "measured" for row in rows) == 24
    assert sum(row["phase"] == "warmup" for row in rows) == 6
    assert "Performance: passed" in (case.run_dir / "report.txt").read_text()
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_performance_failure_keeps_metrics_and_cleans_up_children(
    make_case, simulated_runtime, monkeypatch
):
    monkeypatch.setenv("AFD_E2E_FAKE_NO_USAGE", "1")
    case = make_case(
        *simulated_runtime,
        "--device",
        "0,1,2,3",
        "--test",
        "perf",
        "--perf-mode",
        "mixed",
        "--perf-input-lens",
        "8,32",
        "--perf-output-lens",
        "4",
        "--perf-concurrencies",
        "3",
        "--perf-requests",
        "4",
        "--perf-warmup",
        "0",
    )
    summary = runner.execute(case)
    assert summary["status"] == summary["performance"]["status"] == "failed"
    assert summary["failed_stage"] == "performance"
    point = summary["performance"]["points"][0]
    assert point["completed"] == 0 and point["failed"] == 4
    assert point["metrics"]["tpot_ms"]["mean"] is None
    assert "no token usage" in point["errors"][0]["error"]
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_async_dbo_profiling_advisory_keeps_performance_running(
    make_case, simulated_runtime, monkeypatch
):
    message = (
        "(Worker_DP0_EP0 pid=378826) [ERROR] [378826] profiler.py: "
        + PROFILER_PARSE_ADVISORY
    )
    monkeypatch.setenv("AFD_E2E_FAKE_LIVE_ERROR", message)
    case = make_case(
        *simulated_runtime,
        "--device",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "--model",
        "v4-flash",
        "--model-path",
        "/weights",
        "--comm",
        "async",
        "--mrv",
        "mrv2",
        "--ffn-mrv",
        "mrv1",
        "--dbo",
        "on",
        "--layered-gmm",
        "on",
        "--test",
        "perf",
        "--perf-mode",
        "matrix",
        "--perf-input-lens",
        "8",
        "--perf-output-lens",
        "4",
        "--perf-concurrencies",
        "2",
        "--perf-requests",
        "2",
        "--perf-warmup",
        "0",
    )
    summary = runner.execute(case)
    assert summary["status"] == "passed", summary["errors"]
    assert summary["performance"]["status"] == summary["coverage"]["status"] == "passed"
    assert summary["log_diagnostics"]["ffn"]["inference"] == []
    assert (
        summary["log_advisories"]["ffn"]
        and message in (case.run_dir / "ffn.log").read_text()
    )
    assert (
        json.loads((case.run_dir / "advisories.json").read_text())
        == summary["log_advisories"]
    )
    assert "offline parsing required" in (case.run_dir / "report.txt").read_text()
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_performance_timeout_reads_final_partial_results_after_stopping(
    make_case, simulated_runtime
):
    case = make_case(
        *simulated_runtime,
        "--device",
        "0,1,2,3",
        "--test",
        "perf",
        "--perf-mode",
        "mixed",
        "--perf-input-lens",
        "8",
        "--perf-output-lens",
        "4",
        "--perf-concurrencies",
        "1",
        "--perf-requests",
        "3",
        "--perf-warmup",
        "0",
        "--rps",
        ".1",
        "--eval-timeout",
        "2",
    )
    summary = runner.execute(case)
    assert summary["status"] == "failed" and summary["failed_stage"] == "performance"
    metrics = json.loads((case.run_dir / "performance.json").read_text())
    assert summary["performance"] == metrics
    point = metrics["points"][0]
    assert metrics["status"] == point["status"] == "cancelled"
    assert point["completed"] == 1 and point["unsubmitted"] == 2
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


@pytest.mark.parametrize("comm", ["sync", "async"])
@pytest.mark.parametrize(
    "fault",
    [
        "RuntimeError: FFN loop did not stop",
        "WARNING: force killing remaining processes",
    ],
)
def test_zero_exit_with_shutdown_fault_is_failure(
    make_case, simulated_runtime, monkeypatch, comm, fault
):
    monkeypatch.setenv("AFD_E2E_FAKE_SHUTDOWN_ERROR", fault)
    case = make_case(*simulated_runtime, "--device", "0,1,2,3", "--comm", comm)
    summary = runner.execute(case)
    assert summary["status"] == summary["shutdown_status"] == "failed"
    assert summary["failed_stage"] == "shutdown"
    assert summary["inference_status"] == summary["cleanup_status"] == "passed"
    assert summary["gsm8k"]["strict_match"] == 0.5
    assert summary["process_exit_codes"]["ffn"] == 0
    evidence = summary["log_diagnostics"]["ffn"]["shutdown"][0]
    assert evidence["message"] == fault
    assert (case.run_dir / evidence["log"]).read_text().splitlines()[
        evidence["line"] - 1
    ] == fault
    assert summary["log_diagnostics"]["ffn"]["inference"] == []
    assert all(code is not None for code in summary["process_exit_codes"].values())
    assert (case.run_dir / "exit-code.txt").read_text() == "1\n"
    assert (
        "Inference: passed; shutdown: failed; cleanup: passed"
        in (case.run_dir / "report.txt").read_text()
    )


def test_forced_cleanup_stops_children_but_fails_shutdown(
    make_case, simulated_runtime, monkeypatch
):
    monkeypatch.setenv("AFD_E2E_FAKE_IGNORE_TERM", "1")
    summary = runner.execute(
        make_case(*simulated_runtime, "--device", "0,1,2,3", "--comm", "async")
    )
    assert summary["inference_status"] == summary["cleanup_status"] == "passed"
    assert summary["status"] == summary["shutdown_status"] == "failed"
    assert summary["failed_stage"] == "shutdown"
    assert any("forced SIGKILL" in error for error in summary["errors"])
    assert summary["process_exit_codes"]["ffn"] == -signal.SIGKILL
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_inference_log_fault_aborts_and_cleans_up_children(
    make_case, simulated_runtime, monkeypatch
):
    monkeypatch.setenv("AFD_E2E_FAKE_LIVE_ERROR", "1")
    summary = runner.execute(make_case(*simulated_runtime, "--device", "0,1,2,3"))
    assert summary["status"] == summary["inference_status"] == "failed"
    assert summary["failed_stage"] == "smoke"
    assert any("simulated inference failure" in error for error in summary["errors"])
    assert summary["shutdown_status"] == summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


@pytest.mark.parametrize("comm", ["sync", "async"])
def test_v4_smoke_checks_answers_in_both_modes(
    make_case, simulated_runtime, monkeypatch, comm
):
    monkeypatch.setenv("AFD_E2E_FAKE_WRONG_ANSWER", "1")
    case = make_case(
        *simulated_runtime,
        "--device",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "--model",
        "v4-flash",
        "--model-path",
        "/weights",
        "--comm",
        comm,
    )
    summary = runner.execute(case)
    assert summary["status"] == "failed" and summary["failed_stage"] == "smoke"
    smoke = json.loads((case.run_dir / "smoke.json").read_text())
    assert smoke["answer_checked"]
    assert "Expected 19, got '20'" in smoke["requests"][0]["error"]
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


@pytest.mark.parametrize(
    "failed_file", ["summary.json", "diagnostics.json", "advisories.json", "report.txt"]
)
def test_report_write_failure_cannot_skip_cleanup(
    make_case, simulated_runtime, monkeypatch, failed_file
):
    native_write = Path.write_text

    def fail(path, *args, **kwargs):
        if path.name == failed_file or path.name == f".{failed_file}.tmp":
            raise OSError("simulated full disk")
        return native_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail)
    case = make_case(*simulated_runtime, "--device", "0,1,2,3")
    summary = runner.execute(case)
    assert summary["status"] == "failed"
    assert summary["failed_stage"] == "reporting"
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())
    assert any("simulated full disk" in error for error in summary["errors"])
    assert (case.run_dir / "exit-code.txt").read_text() == "1\n"


def test_progress_is_readable_before_service_is_ready(
    make_case, simulated_runtime, monkeypatch
):
    native_wait = runner.wait_ready
    native_stop = runner._stop_children

    def wait(case, services, progress):
        summary = json.loads((case.run_dir / "summary.json").read_text())
        assert summary["status"] == "running"
        assert summary["stage"] == "startup"
        assert summary["stage_started_at"]
        assert summary["elapsed_seconds"] > 0
        assert "finished_at" not in summary
        return native_wait(case, services, progress)

    def stop(case, *args):
        summary = json.loads((case.run_dir / "summary.json").read_text())
        assert summary["status"] == "running"
        assert summary["stage"] == "cleanup"
        assert summary["inference_status"] == "passed"
        assert "finished_at" not in summary
        return native_stop(case, *args)

    monkeypatch.setattr(runner, "wait_ready", wait)
    monkeypatch.setattr(runner, "_stop_children", stop)
    summary = runner.execute(make_case(*simulated_runtime, "--device", "0,1,2,3"))
    assert summary["status"] == summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_progress_output_error_does_not_interrupt_cleanup(
    make_case, simulated_runtime, monkeypatch
):
    native_print = print

    def fail(*args, **kwargs):
        if args and str(args[0]).startswith("[cleanup]"):
            raise BrokenPipeError("closed progress stream")
        return native_print(*args, **kwargs)

    monkeypatch.setattr(runner, "print", fail, raising=False)
    summary = runner.execute(make_case(*simulated_runtime, "--device", "0,1,2,3"))
    assert summary["status"] == "failed"
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())
    assert any("closed progress stream" in error for error in summary["errors"])


@pytest.mark.parametrize("stage", ["build_services", "wait_ready"])
def test_cancel_during_startup_cleans_up_children(
    make_case, simulated_runtime, monkeypatch, stage
):
    def cancel(*_args):
        raise KeyboardInterrupt("cancelled test")

    monkeypatch.setattr(runner, stage, cancel)
    summary = runner.execute(make_case(*simulated_runtime, "--device", "0,1,2,3"))
    assert summary["status"] == "cancelled"
    assert summary["cleanup_status"] == "passed"
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_surviving_owned_processes_fail_cleanup(
    make_case, simulated_runtime, monkeypatch
):
    monkeypatch.setattr(
        runner,
        "find_processes_matching_environment",
        lambda *_: [type("Identity", (), {"pid": 123})()],
    )
    monkeypatch.setattr(runner, "close_process_identities", lambda *_: None)
    summary = runner.execute(make_case(*simulated_runtime, "--device", "0,1,2,3"))
    assert summary["status"] == "failed"
    assert summary["cleanup_status"] == "failed"
    assert summary["failed_stage"] == "cleanup"
    assert any("Owned processes still alive" in error for error in summary["errors"])
    assert all(code is not None for code in summary["process_exit_codes"].values())


def test_pidfd_cleanup_does_not_kill_other_runs():
    marker = uuid.uuid4().hex
    owned = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env={**os.environ, "AFD_E2E_RUN_ID": marker},
        start_new_session=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env={**os.environ, "AFD_E2E_RUN_ID": "unrelated-" + marker},
        start_new_session=True,
    )
    try:
        assert (
            kill_processes_matching_environment(
                {"AFD_E2E_RUN_ID": marker},
                timeout_s=2,
                poll_interval_s=0.02,
                process_name="test",
            )
            == []
        )
        assert owned.wait(timeout=2) == -signal.SIGKILL
        assert unrelated.poll() is None
    finally:
        for process in (owned, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)


def test_forced_termination_is_failure():
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready'); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        failures = terminate_process_groups(
            [process],
            termination_timeout_s=0.05,
            poll_interval_s=0.01,
            reap_timeout_s=2,
        )
        assert any("forced SIGKILL" in error for error in failures)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        process.stdout.close()


def test_workload_timeout_cleans_up_child(tmp_path):
    child = runner.ManagedProcess(
        "timeout",
        (sys.executable, "-c", "import time; time.sleep(30)"),
        os.environ.copy(),
        tmp_path,
    )
    try:
        with pytest.raises(TimeoutError, match="exceeded"):
            child.wait(0.05, [])
    finally:
        assert (
            terminate_process_groups(
                [child.process],
                termination_timeout_s=2,
                poll_interval_s=0.02,
                reap_timeout_s=2,
            )
            == []
        )
        child.reader.join(timeout=2)
    assert not child.reader.is_alive()
    assert child.process.poll() is not None
