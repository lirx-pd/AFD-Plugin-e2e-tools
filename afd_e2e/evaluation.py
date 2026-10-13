"""Official lm-eval GSM8K task, with optional local Parquet transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .config import Case


class AccuracyError(RuntimeError):
    def __init__(self, result: dict):
        self.result = result
        super().__init__(
            f"GSM8K strict-match {result['strict_match']:.6f} "
            f"< threshold {result['threshold']:.6f}"
        )


def prepare_local_task(dataset: Path, output: Path) -> dict:
    import lm_eval
    import pyarrow.parquet as parquet
    import yaml

    root = dataset / "main" if (dataset / "main").is_dir() else dataset
    data_files = {}
    hashes = {}
    for split, expected_count in (("train", 7473), ("test", 1319)):
        path = (root / f"{split}-00000-of-00001.parquet").resolve(strict=True)
        if parquet.read_metadata(path).num_rows != expected_count:
            raise ValueError(f"Wrong GSM8K {split} row count")
        data_files[split] = str(path)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    original = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml"
    task = yaml.safe_load(original.read_text())
    task.update(
        {
            "dataset_path": "parquet",
            "dataset_name": None,
            "dataset_kwargs": {"data_files": data_files},
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "gsm8k.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    manifest = {
        "data_files": data_files,
        "sha256": hashes,
        "original_task_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    }
    (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def command(case: Case, run_dir: Path) -> tuple[str, ...]:
    host = "127.0.0.1" if case.api_host == "0.0.0.0" else case.api_host
    role = "baseline" if case.baseline else "attention"
    model_args = f"model=afd-e2e-{role},base_url=http://{host}:{case.api_port}/v1/completions,max_gen_toks={case.max_gen_tokens},tokenized_requests=False,num_concurrent={case.concurrency},tokenizer={case.model_path}"
    cmd = [
        case.python,
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        "gsm8k",
        "--num_fewshot",
        str(case.fewshot),
        "--limit",
        str(case.limit),
        "--output_path",
        str(run_dir / "gsm8k"),
        "--log_samples",
        "--seed",
        "42,42,42,42",
    ]
    if case.dataset_path:
        cmd.extend(("--include_path", str(run_dir / "gsm8k-task")))
    return tuple(cmd)


def sample_diagnostics(output: Path, expected_count: int, score: float) -> dict:
    files = list(output.rglob("samples_gsm8k*.jsonl"))
    if len(files) != 1:
        return {"sample_details_unavailable": f"Expected one GSM8K sample log; found {len(files)}"}
    failures, scores, doc_ids = [], [], set()
    mismatch_count = equivalent_count = 0
    try:
        with files[0].open(encoding="utf-8") as stream:
            for line in stream:
                sample = json.loads(line)
                if sample.get("filter") not in {None, "strict-match"}:
                    continue
                match = sample.get("exact_match,strict-match")
                if match is None:
                    match = sample.get("exact_match")
                    if isinstance(match, dict):
                        match = match.get("strict-match")
                    elif sample.get("filter") != "strict-match":
                        continue
                if match not in (0, 1) or sample["doc_id"] in doc_ids:
                    raise ValueError("Invalid or duplicate strict-match sample")
                doc_ids.add(sample["doc_id"])
                scores.append(match)
                if match:
                    continue
                target = str(sample["target"])
                answers = re.findall(r"####\s*([^\r\n]+)", target)
                expected = (answers[-1] if answers else target).strip()
                extracted = sample["filtered_resps"]
                if isinstance(extracted, dict):
                    extracted = extracted["strict-match"]
                while isinstance(extracted, list):
                    extracted = extracted[0] if extracted else ""
                extracted = str(extracted).strip()
                try:
                    numbers = [Decimal(v.replace(",", "").replace("$", ""))
                               for v in (expected, extracted)]
                    equivalent = all(v.is_finite() for v in numbers) and numbers[0] == numbers[1]
                except InvalidOperation:
                    equivalent = False
                mismatch_count += 1
                equivalent_count += int(equivalent)
                if len(failures) < 20:
                    response = sample.get("resps", "")
                    while isinstance(response, list):
                        response = response[0] if response else ""
                    failures.append({
                        "doc_id": sample["doc_id"], "expected": expected,
                        "extracted": extracted, "numeric_equivalent": equivalent,
                        "output_excerpt": str(response)[-400:],
                    })
        if not scores or len(scores) != expected_count or not math.isclose(sum(scores) / len(scores), score):
            raise ValueError("Sample log does not match the official count/score")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"sample_details_unavailable": f"{files[0]}: {exc}"}
    return {
        "sample_file": str(files[0]), "mismatch_count": mismatch_count,
        "numeric_equivalent_mismatches": equivalent_count, "mismatches": failures,
    }


def read_results(output: Path, expected_count: int, threshold: float) -> dict:
    files = list(output.rglob("results_*.json"))
    if not files:
        files = list(output.rglob("results.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one new lm-eval result file; found {len(files)}")
    result = json.loads(files[0].read_text())
    # Flexible extraction is intentionally not an accuracy fallback.
    score = result["results"]["gsm8k"]["exact_match,strict-match"]
    count = result["n-samples"]["gsm8k"]["effective"]
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_count:
        raise RuntimeError(f"Expected {expected_count} GSM8K samples, got {count!r}")
    if (
        isinstance(score, bool)
        or not isinstance(score, (float, int))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise RuntimeError(f"Invalid GSM8K strict-match score: {score!r}")
    metrics = {
        "samples": count,
        "strict_match": score,
        "threshold": threshold,
        "result_file": str(files[0]),
        **sample_diagnostics(output, expected_count, score),
    }
    if score < threshold:
        raise AccuracyError(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_local_task(args.dataset, args.output)), flush=True)


if __name__ == "__main__":
    main()
