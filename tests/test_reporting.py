import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from afd_e2e import cli, reporting, runner
from afd_e2e.reporting import (
    LOG_FAILURE,
    PROFILER_PARSE_ADVISORY,
    is_profiler_advisory,
    pack,
)


@pytest.fixture
def finished_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"status": "passed", "finished_at": "2026-10-11"})
    )
    (run / "attention.log").write_text("inference log\n")
    (run / "gsm8k").mkdir()
    (run / "gsm8k" / "samples.jsonl").write_text('{"answer": "19"}\n')
    (run / "performance.json").write_text('{"status": "passed"}')
    (run / "performance").mkdir()
    (run / "performance" / "requests.jsonl").write_text('{"ttft_ms": 12}\n')
    return run


def test_pack_keeps_evidence_and_inventories_excluded_dumps(finished_run, tmp_path):
    dump = finished_run / "extra-info" / "data-dump" / "npu.bin"
    dump.parent.mkdir(parents=True)
    with dump.open("wb") as stream:
        stream.truncate(3 * 1024**3)
    external = tmp_path / "external.json"
    external.write_text("private external file")
    (finished_run / "external.json").symlink_to(external)
    (finished_run / "external-dir").symlink_to(tmp_path, target_is_directory=True)
    result = pack(finished_run)
    output = Path(result["archive"])
    assert output.stat().st_size < 4096
    assert result["excluded_bytes"] >= 3 * 1024**3
    assert result["included_files"] == 5
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert (
        output.with_name(output.name + ".sha256")
        .read_text()
        .startswith(result["sha256"])
    )
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == {
            "summary.json",
            "attention.log",
            "gsm8k/samples.jsonl",
            "performance.json",
            "performance/requests.jsonl",
            "archive-inventory.json",
        }
        inventory = json.load(archive.extractfile("archive-inventory.json"))
        entry = next(
            item for item in inventory if item["path"] == "extra-info/data-dump/npu.bin"
        )
        assert entry["bytes"] == 3 * 1024**3 and not entry["included"]
    assert dump.exists() and external.read_text() == "private external file"


def test_pack_rejects_a_run_still_in_cleanup(finished_run):
    (finished_run / "summary.json").write_text(
        '{"status": "passed", "stage": "cleanup"}'
    )
    with pytest.raises(ValueError, match="finished"):
        pack(finished_run)
    assert not (finished_run / "results.tar.gz").exists()


def test_pack_never_overwrites_existing_archive(finished_run, tmp_path):
    output = tmp_path / "existing.tar.gz"
    output.write_bytes(b"previous result")
    with pytest.raises(FileExistsError):
        pack(finished_run, output)
    assert output.read_bytes() == b"previous result"


def test_pack_does_not_delete_archive_created_by_another_writer(
    finished_run, tmp_path, monkeypatch
):
    output = tmp_path / "raced.tar.gz"
    native_open = Path.open

    def open_path(path, mode="r", *args, **kwargs):
        if path == output and mode == "xb":
            with native_open(path, "wb") as stream:
                stream.write(b"other writer")
        return native_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(FileExistsError):
        pack(finished_run, output)
    assert output.read_bytes() == b"other writer"


def test_pack_cli_needs_no_devices_or_model_config(finished_run, monkeypatch):
    def unexpected(_args):
        raise AssertionError("pack must not construct an NPU test case")

    monkeypatch.setattr(cli, "from_args", unexpected)
    assert cli.main(["pack", str(finished_run)]) == 0


def test_pack_cleans_its_partial_checksum_on_write_failure(finished_run, monkeypatch):
    native_open = Path.open

    class BadChecksum:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

        def write(self, _text):
            self.stream.write("partial")
            raise OSError("simulated full disk")

    def open_path(path, mode="r", *args, **kwargs):
        stream = native_open(path, mode, *args, **kwargs)
        if path.name.endswith(".sha256") and mode == "x":
            return BadChecksum(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(OSError, match="full disk"):
        pack(finished_run)
    assert not (finished_run / "results.tar.gz").exists()
    assert not (finished_run / "results.tar.gz.sha256").exists()
    assert (finished_run / "attention.log").exists()


@pytest.mark.parametrize(
    "line",
    [
        "(Worker pid=1) ERROR 10-11 03:10:31 WorkerProc hit an exception.",
        "RuntimeError: AFD NPU FFN worker loop did not stop after connector close",
        "Exception: worker failed",
        "Error: worker failed",
        "[rank0]:W1011 RuntimeError: synchronize: error code is 507035",
        "[ERROR] ERR00100 PTA call acl api failed",
        "WARNING [shutdown] Executor: workers still running after SIGTERM; sending SIGKILL count=4",
        "WARNING [shutdown] Process manager: force killing remaining processes count=8",
    ],
)
def test_detects_fault_formats_seen_on_npu(line):
    assert LOG_FAILURE.search(line)


def test_regular_warning_is_not_failure():
    assert not LOG_FAILURE.search(
        "WARNING Triton is unavailable; using PyTorch fallback"
    )


@pytest.mark.parametrize(
    "prefix,suffix,expected",
    [
        ("[ERROR] [378826] profiler.py: ", "", True),
        ("ERROR WorkerProc: ", "", False),
        ("[ERROR] profiler.py: ", " RuntimeError: broken device", False),
        ("RuntimeError: broken device [ERROR] profiler.py: ", "", False),
    ],
)
def test_offline_profiler_parser_advisory_does_not_hide_other_errors(
    prefix, suffix, expected
):
    assert is_profiler_advisory(prefix + PROFILER_PARSE_ADVISORY + suffix) is expected


def test_log_reader_retains_late_root_cause_and_records_profiler_advisory(tmp_path):
    messages = ["[ERROR] profiler.py: " + PROFILER_PARSE_ADVISORY + "\n"]
    messages += [f"ERROR worker loop failed {i}\n" for i in range(25)]
    messages += [f"trace frame {i}\n" for i in range(250)]
    messages += [
        "RuntimeError: call aclnnGroupedMatmulSwigluQuantV2Layered failed, Config_Error(EZ1013)\n"
    ]
    messages += [f"other output {i}\n" for i in range(100)]
    process = object.__new__(runner.ManagedProcess)
    process.name, process.phase = "ffn", "inference"
    process.diagnostics = {"inference": [], "shutdown": []}
    process.advisories, process.reader_error = [], None
    process.process = SimpleNamespace(
        stdout=io.StringIO("".join(messages)), poll=lambda: None
    )
    log = tmp_path / "ffn.log"
    process._read(log)
    errors = process.diagnostics["inference"]
    assert not process.reader_error and len(errors) == 20
    assert [error["line"] for error in errors[:10]] == list(range(2, 12))
    assert "EZ1013" in errors[-1]["message"]
    assert process.advisories[0]["phase"] == "inference"
    assert "profiler.py" not in "".join(error["message"] for error in errors)
    with pytest.raises(RuntimeError, match="worker loop failed"):
        process.ensure_alive()
    evidence = reporting.log_evidence(
        tmp_path,
        {
            "log_diagnostics": {"ffn": process.diagnostics},
            "log_advisories": {"ffn": process.advisories},
        },
    )
    assert (
        evidence["logs"]["ffn"]["sha256"]
        == hashlib.sha256(log.read_bytes()).hexdigest()
    )
    assert evidence["logs"]["ffn"]["lines"] == len(messages)
    assert any("EZ1013" in row["text"] for row in evidence["excerpts"]["ffn"])
    assert evidence["excerpts"]["ffn"][-1]["line"] == len(messages)


def test_compact_pack_includes_context_and_hashes_without_full_logs(finished_run):
    summary_path = finished_run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["log_diagnostics"] = {
        "attention": {
            "inference": [{"line": 200, "message": "RuntimeError: root cause"}],
            "shutdown": [],
        }
    }
    summary_path.write_text(json.dumps(summary))
    log = finished_run / "attention.log"
    lines = [f"frame {i} " + "x" * 300 + "\n" for i in range(1, 1001)]
    lines[199] = "RuntimeError: root cause\n"
    log.write_text("".join(lines))
    original = log.read_bytes()
    result = pack(finished_run, compact=True)
    assert Path(result["archive"]).name == "reports.tar.gz"
    assert result["excluded_bytes"] >= len(original)
    assert Path(result["archive"]).stat().st_size < len(original) / 2
    assert not (finished_run / "log-evidence.json").exists()
    with tarfile.open(result["archive"]) as archive:
        assert "attention.log" not in archive.getnames()
        assert "performance/requests.jsonl" in archive.getnames()
        evidence = json.load(archive.extractfile("log-evidence.json"))
        assert (
            evidence["logs"]["attention"]["sha256"]
            == hashlib.sha256(original).hexdigest()
        )
        rows = {row["line"]: row["text"] for row in evidence["excerpts"]["attention"]}
        assert (
            rows[200] == "RuntimeError: root cause"
            and 192 in rows
            and 240 in rows
            and 1000 in rows
        )
        inventory = json.load(archive.extractfile("archive-inventory.json"))
        assert next(row for row in inventory if row["path"] == "log-evidence.json")[
            "generated"
        ]
    assert log.read_bytes() == original


def test_compact_pack_preserves_previously_downloaded_evidence(
    finished_run, monkeypatch
):
    evidence = b'{"remote_logs":{"ffn":{"sha256":"original"}},"excerpts":{"ffn":[]}}'
    (finished_run / "log-evidence.json").write_bytes(evidence)
    monkeypatch.setattr(
        reporting,
        "log_evidence",
        lambda *_: pytest.fail("must preserve downloaded evidence"),
    )
    result = pack(finished_run, compact=True)
    with tarfile.open(result["archive"]) as archive:
        assert archive.extractfile("log-evidence.json").read() == evidence
    assert (finished_run / "log-evidence.json").read_bytes() == evidence


def test_status_compares_live_and_finished_runs_without_npu(
    tmp_path, monkeypatch, capsys
):
    live, done = tmp_path / "live", tmp_path / "done"
    live.mkdir()
    done.mkdir()
    live_summary = {
        "case_id": "async-dbo-on",
        "status": "running",
        "stage": "gsm8k",
        "elapsed_seconds": 10,
        "inference_status": None,
    }
    done_summary = {
        "case_id": "async-dbo-off",
        "status": "failed",
        "stage": "finished",
        "inference_status": "passed",
        "shutdown_status": "failed",
        "cleanup_status": "passed",
        "gsm8k": {"strict_match": 0.9, "samples": 30},
    }
    (live / "summary.json").write_text(json.dumps(live_summary))
    (done / "summary.json").write_text(json.dumps(done_summary))
    monkeypatch.setattr(
        cli,
        "from_args",
        lambda *_: pytest.fail("status must not configure/allocate NPUs"),
    )
    assert cli.main(["status", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "90.00% (30)" in output and "Shutdown" in output
    assert "gsm8k" in output and "running" in output and "shutdown" in output
    assert "async-dbo-on" in output and "async-dbo-off" in output
    assert cli.main(["status", str(tmp_path), str(live), "--json"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 2
    assert {r["result_dir"] for r in reports} == {str(live), str(done)}
    assert json.loads((done / "summary.json").read_text()) == done_summary


@pytest.mark.parametrize("content", ["{", "[]", '{"status": true}'])
def test_status_rejects_broken_reports_cleanly(tmp_path, capsys, content):
    (tmp_path / "summary.json").write_text(content)
    assert cli.main(["status", str(tmp_path)]) == 2
    assert "afd-e2e:" in capsys.readouterr().err


def test_status_rejects_directory_without_runs(tmp_path, capsys):
    assert cli.main(["status", str(tmp_path)]) == 2
    assert "No run summaries" in capsys.readouterr().err
