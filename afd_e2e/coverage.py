"""Check actual runner and feature execution on every expected role rank."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Case

PREFIX = "AFD_E2E_EVENT "


def analyze(case: Case, run_dir: Path, workload_started: float) -> dict:
    if case.baseline:
        return {
            "status": "not_applicable",
            "reason": "Native baseline has no AFD observers",
        }
    expected = {
        "attention": {
            (dp, tp)
            for dp in range(case.attention_ranks // case.attention_tp)
            for tp in range(case.attention_tp)
        },
        "ffn": {(dp, 0) for dp in range(case.ffn_ranks)},
    }
    initialized = {role: {} for role in expected}
    executed = {role: set() for role in expected}
    graph = {role: set() for role in expected}
    split = {role: set() for role in expected}
    native_ubatches = set()
    layered = set()
    unexpected = []
    for role in expected:
        path = run_dir / f"{role}.log"
        if not path.is_file():
            continue
        with path.open() as stream:
            for line in stream:
                if PREFIX not in line:
                    continue
                event = json.loads(line.split(PREFIX, 1)[1])
                rank = (event["dp_rank"], event["tp_rank"])
                if event["role"] != role or rank not in expected[role]:
                    unexpected.append(f"Unexpected role/rank: {event}")
                    continue
                if event["kind"] == "runner":
                    initialized[role][rank] = event
                if event.get("phase") != "live" or event["time"] < workload_started:
                    continue
                if event["kind"] == "layered":
                    layered.add(rank)
                    continue
                if event["kind"] == "native_ubatches":
                    capacity, real = event["capacity"], event["real"]
                    if (
                        role == "attention"
                        and event["executor"] == "NPUUBatchRunner"
                        and len(capacity) == len(real) == 2
                        and all(c > 0 and 0 <= n <= c for c, n in zip(capacity, real))
                    ):
                        native_ubatches.add(rank)
                    else:
                        unexpected.append(f"Invalid native NPU ubatch trace: {event}")
                    continue
                if event["kind"] != "execution":
                    continue
                executed[role].add(rank)
                if event["mode"] in {"FULL", "replay"}:
                    graph[role].add(rank)
                if event["stages"] == 2:
                    split[role].add(rank)
                    if not case.dbo and not case.moe_ubatching:
                        unexpected.append(
                            f"Unexpected live two-stage execution on {role} {rank}"
                        )
    failures = list(unexpected)
    for role, ranks in expected.items():
        for rank in ranks:
            event = initialized[role].get(rank)
            requested = case.ffn_mrv if role == "ffn" else case.mrv
            if event is None:
                failures.append(f"Missing {role} runner identity on {rank}")
                continue
            if event["mrv_flag"] != int(requested == "mrv2"):
                failures.append(f"Wrong {role} MRV flag on {rank}")
            if role == "ffn":
                expected_class = "AFDNPUFFNModelRunner"
            elif requested == "mrv2":
                expected_class = "AFDNPUAttentionModelRunnerV2"
            else:
                expected_class = "AFDNPUAttentionModelRunner"
            if event["class"].rsplit(".", 1)[-1] != expected_class:
                failures.append(
                    f"Wrong {role} runner class on {rank}: {event['class']}"
                )
            if role == "ffn" and event.get("layered") != case.layered_gmm:
                failures.append(f"Wrong actual Layered GMM state on {rank}")
        if missing := ranks - executed[role]:
            failures.append(f"Missing live {role} execution on {sorted(missing)}")
        if case.execution == "graph" and (missing := ranks - graph[role]):
            failures.append(f"Missing live {role} graph replay on {sorted(missing)}")
        require_split = (case.dbo and (case.comm == "sync" or role == "attention")) or (
            case.moe_ubatching and role == "attention"
        )
        if require_split and (missing := ranks - split[role]):
            failures.append(
                f"Missing live {role} two-stage execution on {sorted(missing)}"
            )
    if case.layered_gmm and (missing := expected["ffn"] - layered):
        failures.append(f"Missing actual Layered GMM calls on {sorted(missing)}")
    if (
        case.dbo or (case.comm == "async" and case.moe_ubatching)
    ) and case.mrv == "mrv2":
        if missing := expected["attention"] - native_ubatches:
            failures.append(f"Missing native NPU ubatch execution on {sorted(missing)}")
    report = {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "runner_identities": {
            role: list(events.values()) for role, events in initialized.items()
        },
        "live_ranks": {role: sorted(ranks) for role, ranks in executed.items()},
        "graph_ranks": {role: sorted(ranks) for role, ranks in graph.items()},
        "two_stage_ranks": {role: sorted(ranks) for role, ranks in split.items()},
        "native_ubatch_ranks": sorted(native_ubatches),
        "layered_ranks": sorted(layered),
    }
    (run_dir / "coverage.json").write_text(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("Execution coverage failed: " + "; ".join(failures))
    return report
