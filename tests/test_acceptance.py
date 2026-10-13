import json

import pytest

from afd_e2e import evaluation


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


@pytest.mark.parametrize("schema", ["per-filter", "combined"])
def test_formatting_diagnostics_do_not_change_strict_accuracy(tmp_path, schema):
    result = {
        "results": {"gsm8k": {"exact_match,strict-match": 1 / 3}},
        "n-samples": {"gsm8k": {"effective": 3}},
    }
    (tmp_path / "results_1.json").write_text(json.dumps(result))
    samples = []
    for doc_id, (expected, actual, match) in enumerate(
        [("26", "26.00", 0), ("13", "12", 0), ("19", "19", 1)]
    ):
        row = {
            "doc_id": doc_id, "target": f"Solution\n#### {expected}",
            "resps": [[f"Answer: #### {actual}"]],
        }
        if schema == "per-filter":
            row.update(filter="strict-match", exact_match=match, filtered_resps=[actual])
            samples.append({**row, "filter": "flexible-extract", "exact_match": 1})
        else:
            row.update(exact_match={"strict-match": match}, filtered_resps={"strict-match": [actual]})
        samples.append(row)
    (tmp_path / "samples_gsm8k_1.jsonl").write_text(
        "\n".join(json.dumps(row) for row in samples)
    )
    with pytest.raises(evaluation.AccuracyError) as failure:
        evaluation.read_results(tmp_path, 3, 0.5)
    metrics = failure.value.result
    assert metrics["strict_match"] == 1 / 3
    assert metrics["mismatch_count"] == 2
    assert metrics["numeric_equivalent_mismatches"] == 1
    assert [row["numeric_equivalent"] for row in metrics["mismatches"]] == [True, False]
    assert metrics["mismatches"][0]["expected"] == "26"
    assert metrics["mismatches"][0]["extracted"] == "26.00"
    assert "26.00" in metrics["mismatches"][0]["output_excerpt"]


def test_broken_sample_log_cannot_hide_accuracy_failure(tmp_path):
    (tmp_path / "results_1.json").write_text(json.dumps({
        "results": {"gsm8k": {"exact_match,strict-match": 0.1}},
        "n-samples": {"gsm8k": {"effective": 3}},
    }))
    (tmp_path / "samples_gsm8k_1.jsonl").write_text("{broken")
    with pytest.raises(evaluation.AccuracyError) as failure:
        evaluation.read_results(tmp_path, 3, 0.5)
    assert failure.value.result["strict_match"] == 0.1
    assert failure.value.result["sample_details_unavailable"]
