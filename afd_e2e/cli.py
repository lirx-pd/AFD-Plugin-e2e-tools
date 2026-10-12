"""One local entry point, with optional in-Pod device manager commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import from_args
from .planner import describe
from .runner import execute, exit_code, write_json
from .reporting import pack, read_status, status_text


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Portable AFD NPU E2E tests; afd-plugin is a sibling directory"
    )
    subparsers = root.add_subparsers(dest="command", required=True)
    archive = subparsers.add_parser(
        "pack", help="Pack finished run logs without NPU dumps"
    )
    archive.add_argument("run_dir", type=Path)
    archive.add_argument("--output", type=Path, help="New archive path")
    archive.add_argument(
        "--compact",
        action="store_true",
        help="Keep reports and error context; omit full service logs",
    )
    status = subparsers.add_parser(
        "status", help="Read one run or compare a directory of runs; no NPU needed"
    )
    status.add_argument(
        "paths", type=Path, nargs="+", help="Run directories or their parent directory"
    )
    status.add_argument(
        "--json", action="store_true", help="Output complete summaries as JSON"
    )
    for name in ("plan", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--model", choices=("v2-lite", "v4-flash"), default="v2-lite")
        sub.add_argument(
            "--model-path", help="Local checkpoint or HF model ID; required for V4"
        )
        sub.add_argument(
            "--repo",
            type=Path,
            help="Defaults to afd-plugin beside this tool directory",
        )
        sub.add_argument("--topology", choices=("2a1f", "2a2f", "dp2tp4-ep8"))
        sub.add_argument("--comm", choices=("sync", "async"), default="sync")
        sub.add_argument("--mrv", choices=("mrv1", "mrv2"), default="mrv1")
        sub.add_argument(
            "--ffn-mrv", choices=("mrv1", "mrv2"), help="Overrides FFN's MRV flag"
        )
        sub.add_argument("--execution", choices=("eager", "graph"), default="eager")
        sub.add_argument(
            "--dbo", choices=("on", "off"), default="off", help="Native vLLM DBO"
        )
        sub.add_argument("--layered-gmm", choices=("on", "off"), default="off")
        sub.add_argument(
            "--moe-ubatching",
            choices=("on", "off"),
            help="Async CAM token splitting; defaults on for async",
        )
        sub.add_argument(
            "--devices",
            "--device",
            help="Comma-separated IDs, e.g. 0,1,2; no platform file required",
        )
        sub.add_argument(
            "--npu-count",
            type=int,
            help="Requested device count; defaults to topology size",
        )
        sub.add_argument(
            "--platform",
            type=Path,
            help="Optional YAML/JSON device-manager commands, not a Pod manager",
        )
        sub.add_argument(
            "--allocate-command",
            help="Manager command; {count} and {run_id} placeholders",
        )
        sub.add_argument(
            "--release-command",
            help="Manager command; {lease_id} and {run_id} placeholders",
        )
        sub.add_argument(
            "--limit", default="30", help="Exact GSM8K sample count or all (1319)"
        )
        sub.add_argument(
            "--test", choices=("smoke", "gsm8k", "all", "perf"), default="all"
        )
        sub.add_argument(
            "--perf-input-lens",
            default="128,512,2048",
            help="Input token lengths, comma-separated",
        )
        sub.add_argument(
            "--perf-output-lens",
            default="128",
            help="Output token lengths, comma-separated",
        )
        sub.add_argument(
            "--perf-concurrencies",
            default="1,8,32",
            help="Client in-flight limits, comma-separated",
        )
        sub.add_argument(
            "--rps",
            "--perf-rps",
            dest="perf_rps",
            default="inf",
            help="Arrival rates, comma-separated; inf saturates the concurrency limit",
        )
        sub.add_argument(
            "--perf-mode", choices=("matrix", "mixed", "both"), default="both"
        )
        sub.add_argument(
            "--perf-weights",
            help="Mixed input-length weights, e.g. 8,1,1; default equal",
        )
        sub.add_argument(
            "--perf-arrival",
            choices=("constant", "poisson", "burst"),
            default="constant",
        )
        sub.add_argument(
            "--perf-requests", type=int, default=100, help="Measured requests per point"
        )
        sub.add_argument(
            "--perf-warmup",
            type=int,
            default=3,
            help="Unmeasured warmup requests per point",
        )
        sub.add_argument(
            "--dataset-path",
            type=Path,
            help="Local GSM8K directory with main/train and test Parquet files",
        )
        sub.add_argument("--threshold", type=float, default=0.27)
        sub.add_argument("--fewshot", type=int, default=8)
        sub.add_argument("--concurrency", type=int, default=12)
        sub.add_argument("--max-gen-tokens", type=int, default=512)
        sub.add_argument("--max-model-len", type=int)
        sub.add_argument("--max-batched-tokens", type=int)
        sub.add_argument("--max-seqs", type=int)
        sub.add_argument("--memory-utilization", type=float, default=0.7)
        sub.add_argument("--capture-sizes", default="1,2,4,8,16")
        sub.add_argument("--dynamic-quant", type=int, choices=(0, 1))
        sub.add_argument("--api-host", default="127.0.0.1")
        sub.add_argument("--api-port", type=int, default=19280)
        sub.add_argument(
            "--afd-host",
            help="Advertised IP; otherwise HCCL_IF_IP or the default route interface",
        )
        sub.add_argument("--afd-port", type=int, default=6455)
        sub.add_argument(
            "--nic", help="Communication interface; otherwise inherited or detected"
        )
        sub.add_argument(
            "--python", help="Target Python; defaults to the current interpreter"
        )
        sub.add_argument(
            "--vllm-bin", default="vllm", help="vLLM CLI in the target environment"
        )
        sub.add_argument(
            "--offline",
            action="store_true",
            help="Use only local/cached model and dataset files",
        )
        sub.add_argument("--startup-timeout", type=float, default=1800)
        sub.add_argument("--eval-timeout", type=float, default=28800)
        sub.add_argument("--request-timeout", type=float, default=300)
        sub.add_argument("--cleanup-timeout", type=float, default=60)
        sub.add_argument(
            "--output",
            type=Path,
            help="New output directory; defaults to runs/<unique-id>",
        )
        sub.add_argument(
            "--baseline",
            action="store_true",
            help="Native V2-Lite baseline on 4 devices",
        )
        sub.add_argument(
            "--vllm-arg",
            action="append",
            default=[],
            help="Extra token; use --vllm-arg=--flag",
        )
        sub.add_argument("--attention-arg", action="append", default=[])
        sub.add_argument("--ffn-arg", action="append", default=[])
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "pack":
            print(
                json.dumps(
                    pack(args.run_dir, args.output, compact=args.compact),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "status":
            reports = read_status(args.paths)
            print(
                json.dumps(reports, ensure_ascii=False, indent=2)
                if args.json
                else status_text(reports),
                end="\n" if args.json else "",
            )
            return 0
        case = from_args(args)
        if args.command == "plan":
            result = describe(case)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if case.run_dir:
                case.run_dir.mkdir(parents=True, exist_ok=False)
                write_json(case.run_dir / "plan.json", result)
            return 0
        report = execute(case)
        return exit_code(report["status"])
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"afd-e2e: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
