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
