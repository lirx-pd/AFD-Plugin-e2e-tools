"""Readable results and portable archives of the harness-owned evidence."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections import deque
from pathlib import Path


# shortcut: extend these fatal log formats when adopting a platform with other formats.
LOG_FAILURE = re.compile(
    r"\b\w*(?:Error|Exception):|\[ERROR\]|\bERROR\s"
    r"|force killing|sending SIGKILL|forced SIGKILL"
)
PROFILER_PARSE_ADVISORY = (
    "The profiling data cannot be parsed during the daemon process, "
    "it is recommended that you use an offline parsing interface to parse the collected data."
)


def is_profiler_advisory(line: str) -> bool:
    # Offline export is required; this message does not report an inference failure.
    return (
        "profiler.py:" in line
        and line.rstrip().endswith(PROFILER_PARSE_ADVISORY)
        and len(LOG_FAILURE.findall(line)) == 1
    )


def log_evidence(run_dir: Path, summary: dict) -> dict:
    result = {"logs": {}, "excerpts": {}}
    for role in ("attention", "ffn"):
        path = run_dir / f"{role}.log"
        if not path.is_file() or path.is_symlink():
            continue
        errors = [
            error
            for phase in summary.get("log_diagnostics", {}).get(role, {}).values()
            for error in phase
        ] + summary.get("log_advisories", {}).get(role, [])
        positions = {
            number
            for error in errors
            for number in range(max(1, error["line"] - 8), error["line"] + 41)
        }
        digest, size, count = hashlib.sha256(), 0, 0
        excerpts, tail = [], deque(maxlen=40)
        limit = path.stat().st_size
        with path.open("rb") as stream:
            while size < limit:
                line = stream.readline(limit - size)
                if not line:
                    break
                count += 1
                digest.update(line)
                size += len(line)
                row = {
                    "line": count,
                    "text": line.decode("utf-8", errors="replace").rstrip("\r\n"),
                }
                tail.append(row)
                if count in positions:
                    excerpts.append(row)
        rows = {row["line"]: row for row in (*excerpts, *tail)}
        result["logs"][role] = {
            "path": path.name,
            "bytes": size,
            "lines": count,
            "sha256": digest.hexdigest(),
        }
        result["excerpts"][role] = [rows[number] for number in sorted(rows)]
    return result


def report_text(summary: dict) -> str:
    lines = [
        f"Result: {summary['status']}",
        f"Case: {summary['case_id']}",
        "Inference: {inference_status}; shutdown: {shutdown_status}; cleanup: {cleanup_status}".format(
            **summary
        ),
        f"Elapsed: {summary['elapsed_seconds']:.1f}s",
        f"Output: {summary['run_dir']}",
    ]
    if score := summary.get("gsm8k"):
        lines.append(
            f"GSM8K: {score['strict_match']:.2%}, samples={score['samples']}, "
            f"threshold={score['threshold']:.2%}"
        )
    if evidence := summary.get("coverage"):
        lines.append(f"Coverage: {evidence['status']}")
    if performance := summary.get("performance"):
        lines.append(
            f"Performance: {performance['status']} (client ms; see performance.json)"
        )
        for point in performance["points"]:
            metrics = point.get("metrics", {})
            lines.append(
                f"{point['id']} {point.get('phase', 'end-to-end')} {point['mode']} "
                f"input={point['input_lens']} output={point['output_lens']} "
                f"C={point['concurrency']} RPS={point['rps'] if point['rps'] is not None else 'inf'} "
                f"passed={point.get('completed', 0)}/{point['requests']}"
            )
            for name in ("ttft", "tpot"):
                values = [
                    metrics.get(name + "_ms", {}).get(key)
                    for key in ("mean", "p50", "p95", "p99")
                ]
                formatted = "/".join(
                    f"{v:.3f}" if v is not None else "-" for v in values
                )
                lines[-1] += f" {name.upper()} mean/p50/p95/p99={formatted}"
            rate = point.get("achieved_rps")
            lines[-1] += f" completed RPS={rate:.3f}" if rate is not None else ""
    if summary.get("failed_stage"):
        lines.append(f"Failed stage: {summary['failed_stage']}")
    lines.extend(f"Error: {error}" for error in summary["errors"])
    if any(summary.get("log_advisories", {}).values()):
        lines.append("Profiler: offline parsing required; see advisories.json")
    if any(
        evidence
        for phases in summary.get("log_diagnostics", {}).values()
        for evidence in phases.values()
    ):
        lines.append(
            "Log evidence: diagnostics.json; pack --compact includes log context and hashes"
        )
    return "\n".join(lines) + "\n"


def read_status(paths: list[Path]) -> list[dict]:
    reports = {}
    for path in paths:
        files = (
            [path / "summary.json"]
            if (path / "summary.json").is_file()
            else sorted(path.glob("*/summary.json"))
        )
        if not files:
            raise ValueError(f"No run summaries found in {path}")
        for file in files:
            directory = str(file.parent.resolve())
            summary = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(summary, dict) or not isinstance(
                summary.get("status"), str
            ):
                raise ValueError(f"Invalid run summary: {file}")
            reports[directory] = {**summary, "result_dir": directory}
    return list(reports.values())


def status_text(reports: list[dict]) -> str:
    rows = [
        [
            "Run",
            "Case",
            "Result",
            "Stage",
            "Inference",
            "Shutdown",
            "Cleanup",
            "Coverage",
            "GSM8K",
            "Perf",
            "Elapsed",
        ]
    ]
    for report in reports:
        score = report.get("gsm8k")
        elapsed = report.get("elapsed_seconds")
        stage = report.get("failed_stage")
        if not stage and report["status"] == "failed":
            stage = next(
                (
                    phase
                    for phase in ("inference", "shutdown", "cleanup")
                    if report.get(phase + "_status") == "failed"
                ),
                None,
            )
        rows.append(
            [
                Path(report["result_dir"]).name,
                report.get("case_id") or "-",
                report["status"],
                stage or report.get("stage") or "-",
                *(
                    report.get(name) or "-"
                    for name in (
                        "inference_status",
                        "shutdown_status",
                        "cleanup_status",
                    )
                ),
                (report.get("coverage") or {}).get("status") or "-",
                f"{score['strict_match']:.2%} ({score['samples']})" if score else "-",
                (report.get("performance") or {}).get("status") or "-",
                f"{elapsed:.1f}s" if elapsed is not None else "-",
            ]
        )
    widths = [max(len(value) for value in column) for column in zip(*rows)]
    return (
        "\n".join(
            "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
            for row in rows
        )
        + "\n"
    )


def pack(run_dir: Path, output: Path | None = None, *, compact: bool = False) -> dict:
    run_dir = run_dir.resolve(strict=True)
    summary = json.loads((run_dir / "summary.json").read_text())
    if not summary.get("finished_at") or summary.get("status") not in {
        "passed",
        "failed",
        "cancelled",
    }:
        raise ValueError("Can only pack a finished run")
    output = (
        output.resolve()
        if output
        else run_dir / ("reports.tar.gz" if compact else "results.tar.gz")
    )
    checksum_path = output.with_name(output.name + ".sha256")
    if output.exists() or checksum_path.exists():
        raise FileExistsError(f"Archive or checksum already exists: {output}")
    included = []
    inventory = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        owned_output = (
            len(relative.parts) == 1 and path.suffix in {".json", ".log", ".txt"}
        ) or (
            relative.parts[0] in {"gsm8k", "gsm8k-task", "performance"}
            and path.suffix in {".json", ".jsonl", ".yaml"}
        )
        include = (
            owned_output
            and not (compact and relative.name in {"attention.log", "ffn.log"})
            and not path.is_symlink()
            and path.resolve().is_relative_to(run_dir)
        )
        inventory.append(
            {
                "path": str(relative),
                "bytes": path.lstat().st_size,
                "included": include,
            }
        )
        if include:
            included.append(path)
    generated = None
    if compact and run_dir / "log-evidence.json" not in included:
        generated = json.dumps(
            log_evidence(run_dir, summary), ensure_ascii=False, indent=2
        ).encode()
        inventory.append(
            {
                "path": "log-evidence.json",
                "bytes": len(generated),
                "included": True,
                "generated": True,
            }
        )
    contents = json.dumps(inventory, ensure_ascii=False, indent=2).encode()
    created = False
    checksum_created = False
    try:
        with output.open("xb") as destination:
            created = True
            with tarfile.open(fileobj=destination, mode="w:gz") as archive:
                for path in included:
                    archive.add(
                        path, arcname=str(path.relative_to(run_dir)), recursive=False
                    )
                if generated is not None:
                    entry = tarfile.TarInfo("log-evidence.json")
                    entry.size = len(generated)
                    archive.addfile(entry, io.BytesIO(generated))
                entry = tarfile.TarInfo("archive-inventory.json")
                entry.size = len(contents)
                archive.addfile(entry, io.BytesIO(contents))
        digest = hashlib.sha256()
        with output.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        with checksum_path.open("x") as stream:
            checksum_created = True
            stream.write(f"{digest.hexdigest()}  {output.name}\n")
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        if checksum_created:
            checksum_path.unlink(missing_ok=True)
        raise
    return {
        "archive": str(output),
        "sha256": digest.hexdigest(),
        "included_files": len(included) + int(generated is not None),
        "compact": compact,
        "excluded_files": sum(not item["included"] for item in inventory),
        "excluded_bytes": sum(
            item["bytes"] for item in inventory if not item["included"]
        ),
    }
