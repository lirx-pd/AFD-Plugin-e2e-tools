import json

import pytest

from afd_e2e import coverage, evaluation


def write_events(case, path, *, time=20, mode="eager", stages=0, layered_calls=True):
    for role, ranks, tp in (
        ("attention", case.attention_ranks, case.attention_tp),
        ("ffn", case.ffn_ranks, 1),
    ):
        events = []
        for dp in range(ranks // tp):
            for rank in range(tp):
                common = {"role": role, "dp_rank": dp, "tp_rank": rank, "time": time}
                requested = case.ffn_mrv if role == "ffn" else case.mrv
                name = (
                    "AFDNPUFFNModelRunner"
                    if role == "ffn"
                    else "AFDNPUAttentionModelRunnerV2"
                    if requested == "mrv2"
                    else "AFDNPUAttentionModelRunner"
                )
                events.append(
                    {
                        **common,
                        "kind": "runner",
                        "class": name,
                        "mrv_flag": int(requested == "mrv2"),
                        "layered": case.layered_gmm,
                    }
                )
                events.append(
                    {
                        **common,
                        "kind": "execution",
                        "phase": "live",
                        "mode": mode,
                        "stages": stages,
                    }
                )
                if role == "ffn" and case.layered_gmm and layered_calls:
                    events.append({**common, "kind": "layered", "phase": "live"})
        (path / f"{role}.log").write_text(
            "".join(coverage.PREFIX + json.dumps(e) + "\n" for e in events)
        )


def test_graph_requires_live_replay(make_case, tmp_path):
    case = make_case("--execution", "graph")
    write_events(case, tmp_path, mode="FULL")
    assert coverage.analyze(case, tmp_path, 10)["status"] == "passed"
    write_events(case, tmp_path, mode="eager")
    with pytest.raises(RuntimeError, match="graph replay"):
        coverage.analyze(case, tmp_path, 10)
    write_events(case, tmp_path, mode="FULL", time=9)
    with pytest.raises(RuntimeError, match="Missing live"):
        coverage.analyze(case, tmp_path, 10)


def test_every_rank_must_execute(make_case, tmp_path):
    case = make_case()
    write_events(case, tmp_path)
    log = tmp_path / "ffn.log"
    log.write_text(
        "\n".join(
            line for line in log.read_text().splitlines() if '"dp_rank": 1' not in line
        )
    )
    with pytest.raises(RuntimeError, match="Missing live ffn execution"):
        coverage.analyze(case, tmp_path, 10)


def test_split_both_positive_and_negative(make_case, tmp_path):
    on = make_case("--dbo", "on")
    write_events(on, tmp_path, stages=0)
    with pytest.raises(RuntimeError, match="two-stage"):
        coverage.analyze(on, tmp_path, 10)
    write_events(on, tmp_path, stages=2)
    assert coverage.analyze(on, tmp_path, 10)["status"] == "passed"
    with pytest.raises(RuntimeError, match="Unexpected live two-stage"):
        coverage.analyze(make_case(), tmp_path, 10)


def test_mrv2_dbo_requires_actual_native_executor_trace(make_case, tmp_path):
    case = make_case("--mrv", "mrv2", "--ffn-mrv", "mrv1", "--dbo", "on")
    write_events(case, tmp_path, stages=2)
    with pytest.raises(RuntimeError, match="native NPU ubatch"):
        coverage.analyze(case, tmp_path, 10)
    with (tmp_path / "attention.log").open("a") as stream:
        for dp in range(2):
            event = {
                "role": "attention",
                "dp_rank": dp,
                "tp_rank": 0,
                "time": 20,
                "kind": "native_ubatches",
                "phase": "live",
                "executor": "NPUUBatchRunner",
                "capacity": [4, 4],
                "real": [4, 1],
            }
            stream.write(coverage.PREFIX + json.dumps(event) + "\n")
    assert coverage.analyze(case, tmp_path, 10)["native_ubatch_ranks"] == [
        (0, 0),
        (1, 0),
    ]


def test_layered_flag_is_insufficient_without_calls(make_case, tmp_path):
    case = make_case(
        "--model",
        "v4-flash",
        "--model-path",
        "/weights",
        "--comm",
        "async",
        "--layered-gmm",
        "on",
    )
    write_events(case, tmp_path, stages=2, layered_calls=False)
    with pytest.raises(RuntimeError, match="actual Layered GMM calls"):
        coverage.analyze(case, tmp_path, 10)
    write_events(case, tmp_path, stages=2)
    assert len(coverage.analyze(case, tmp_path, 10)["layered_ranks"]) == 8


@pytest.mark.parametrize(
    "score,count,error",
    [
        (0.5, 7, None),
        (0.1, 7, "threshold"),
        (0.5, 6, "samples"),
        (float("nan"), 7, "Invalid"),
        (True, 7, "Invalid"),
        (0.5, True, "samples"),
    ],
)
def test_strict_accuracy_and_exact_count(tmp_path, score, count, error):
    result = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": score,
                "exact_match,flexible-extract": 1,
            }
        },
        "n-samples": {"gsm8k": {"effective": count}},
    }
    (tmp_path / "results_1.json").write_text(json.dumps(result))
    if error:
        with pytest.raises(RuntimeError, match=error):
            evaluation.read_results(tmp_path, 7, 0.27)
    else:
        assert evaluation.read_results(tmp_path, 7, 0.27)["strict_match"] == 0.5


def test_stale_or_ambiguous_results_fail(tmp_path):
    for name in ("results_a.json", "results_b.json"):
        (tmp_path / name).write_text("{}")
    with pytest.raises(RuntimeError, match="found 2"):
        evaluation.read_results(tmp_path, 7, 0.27)
