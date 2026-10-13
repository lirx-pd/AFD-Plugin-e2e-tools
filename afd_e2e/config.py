"""CPU-safe configuration and current NPU runtime constraints."""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

FULL_GSM8K_COUNT = 1319
TOOL_ROOT = Path(__file__).resolve().parents[1]


def parse_devices(value: str | list) -> tuple[str, ...]:
    if not isinstance(value, (str, list, tuple)):
        raise ValueError(
            "Devices must be a comma-separated string or list of integer IDs"
        )
    parts = value.split(",") if isinstance(value, str) else value
    devices = tuple(str(part).strip() for part in parts)
    if not devices or any(
        not device.isascii() or not device.isdecimal() for device in devices
    ):
        raise ValueError("Devices must be comma-separated nonnegative integer IDs")
    devices = tuple(str(int(device)) for device in devices)
    if len(devices) != len(set(devices)):
        raise ValueError("Device IDs must be unique")
    return devices


def default_repo() -> Path:
    candidates = (
        TOOL_ROOT.parent / "afd-plugin",
        Path.cwd() / "afd-plugin",
        Path.cwd().parent / "afd-plugin",
    )
    return next(
        (p.resolve() for p in candidates if (p / "afd_plugin").is_dir()),
        candidates[0].resolve(),
    )


def baseline_layouts(topology: str, model: str) -> tuple[tuple[int, int], ...]:
    layouts = []
    for item in topology.split(","):
        match = re.fullmatch(r"dp([1-9][0-9]*)(?:tp([1-9][0-9]*))?", item.strip())
        if match is None:
            raise ValueError("Baseline topology requires dpN or dpNtpM, comma-separated")
        dp, tp = match.groups()
        layouts.append((int(dp), int(tp) if tp else (4 if model == "v4-flash" else 1)))
    return tuple(layouts)


@dataclass(frozen=True)
class PerformanceConfig:
    input_lens: tuple[int, ...]
    output_lens: tuple[int, ...]
    concurrencies: tuple[int, ...]
    rps: tuple[float | None, ...]
    weights: tuple[float, ...]
    requests: int
    warmup: int
    mode: str
    arrival: str
    phase: str


def performance_configuration(args) -> PerformanceConfig:
    input_default = "1" if args.perf_phase == "decode" else "128,512,2048"
    output_default = "1" if args.perf_phase == "prefill" else "128"
    lengths = {
        name: tuple(int(value) for value in raw.split(","))
        for name, raw in (
            (
                "input_lens",
                args.perf_input_lens
                if args.perf_input_lens is not None
                else input_default,
            ),
            (
                "output_lens",
                args.perf_output_lens
                if args.perf_output_lens is not None
                else output_default,
            ),
            ("concurrencies", args.perf_concurrencies),
        )
    }
    for name, values in lengths.items():
        if not values or min(values) <= 0 or len(set(values)) != len(values):
            raise ValueError(
                f"perf-{name.replace('_', '-')} requires unique positive integers"
            )
    if args.perf_phase == "prefill" and lengths["output_lens"] != (1,):
        raise ValueError("--perf-phase prefill requires --perf-output-lens 1")
    if args.perf_phase == "decode":
        if lengths["input_lens"] != (1,):
            raise ValueError("--perf-phase decode requires --perf-input-lens 1")
        if min(lengths["output_lens"]) <= 1:
            raise ValueError("--perf-phase decode requires output lengths greater than 1")
    rates = tuple(
        None if value.strip().lower() == "inf" else float(value)
        for value in args.perf_rps.split(",")
    )
    if len(set(rates)) != len(rates) or any(
        rate is not None and (not math.isfinite(rate) or rate <= 0) for rate in rates
    ):
        raise ValueError("--rps requires unique positive finite rates, or inf")
    weights = (
        tuple(float(value) for value in args.perf_weights.split(","))
        if args.perf_weights
        else (1.0,) * len(lengths["input_lens"])
    )
    if (
        len(weights) != len(lengths["input_lens"])
        or any(not math.isfinite(w) or w <= 0 for w in weights)
        or not math.isfinite(sum(weights))
    ):
        raise ValueError(
            "--perf-weights needs one positive finite weight per input length"
        )
    if args.perf_requests <= 0 or args.perf_warmup < 0:
        raise ValueError(
            "perf-requests must be positive; perf-warmup must be nonnegative"
        )
    return PerformanceConfig(
        **lengths,
        rps=rates,
        weights=weights,
        requests=args.perf_requests,
        warmup=args.perf_warmup,
        mode=args.perf_mode,
        arrival=args.perf_arrival,
        phase=args.perf_phase,
    )


@dataclass(frozen=True)
class Case:
    model: str
    model_path: str
    repo: Path
    topology: str
    comm: str
    mrv: str
    ffn_mrv: str
    execution: str
    dbo: bool
    layered_gmm: bool
    moe_ubatching: bool
    devices: tuple[str, ...] | None
    limit: int
    test: str
    performance: PerformanceConfig
    dataset_path: Path | None
    threshold: float
    fewshot: int
    concurrency: int
    max_gen_tokens: int
    max_model_len: int
    max_batched_tokens: int
    max_seqs: int
    memory_utilization: float
    capture_sizes: tuple[int, ...]
    dynamic_quant: int
    api_host: str
    api_port: int
    afd_host: str | None
    afd_port: int
    nic: str | None
    python: str
    vllm_bin: str
    offline: bool
    startup_timeout: float
    eval_timeout: float
    request_timeout: float
    cleanup_timeout: float
    run_dir: Path | None
    baseline: bool
    common_args: tuple[str, ...]
    attention_args: tuple[str, ...]
    ffn_args: tuple[str, ...]
    omit_args: tuple[str, ...]
    serve_env: tuple[str, ...]

    @property
    def tests(self) -> tuple[str, ...]:
        selected = {name.strip() for name in self.test.split(",")}
        if not selected <= {"smoke", "gsm8k", "perf", "all"}:
            raise ValueError("--test accepts smoke, gsm8k, perf or all, comma-separated")
        if "all" in selected:
            selected.update(("smoke", "gsm8k"))
        return tuple(name for name in ("smoke", "gsm8k", "perf") if name in selected)

    @property
    def attention_ranks(self) -> int:
        if self.baseline:
            return sum(dp * tp for dp, tp in self.baseline_layouts)
        return 8 if self.model == "v4-flash" else 2

    @property
    def ffn_ranks(self) -> int:
        if self.baseline:
            return 0
        if self.model == "v4-flash":
            return 8
        return 1 if self.topology == "2a1f" else 2

    @property
    def required_devices(self) -> int:
        return self.attention_ranks + self.ffn_ranks

    @property
    def attention_tp(self) -> int:
        if self.baseline:
            return self.baseline_layouts[0][1]
        if self.model == "v4-flash":
            return 4
        return 2 if self.comm == "async" else 1

    @property
    def case_id(self) -> str:
        if self.baseline:
            return f"{self.model}-baseline-{self.topology}-{self.mrv}-{self.execution}"
        return "-".join(
            (
                self.model,
                self.topology,
                self.comm,
                self.mrv,
                f"ffn-{self.ffn_mrv}",
                self.execution,
                f"dbo-{int(self.dbo)}",
                f"moe-{int(self.moe_ubatching)}",
                f"layered-{int(self.layered_gmm)}",
            )
        )

    @property
    def baseline_layouts(self) -> tuple[tuple[int, int], ...]:
        return baseline_layouts(self.topology, self.model)

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in ("repo", "dataset_path", "run_dir"):
            result[key] = str(result[key]) if result[key] is not None else None
        result["case_id"] = self.case_id
        result["required_devices"] = self.required_devices
        result["tests"] = list(self.tests)
        return result


def device_configuration(args) -> tuple[str, ...] | None:
    devices = parse_devices(args.devices) if args.devices is not None else None
    inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if (
        devices is not None
        and inherited
        and not set(devices) <= set(parse_devices(inherited))
    ):
        raise ValueError(
            "Requested devices are outside inherited ASCEND_RT_VISIBLE_DEVICES"
        )
    if devices is None:
        devices = parse_devices(inherited) if inherited else None
    return devices


def from_args(args) -> Case:
    devices = device_configuration(args)
    model_path = args.model_path
    if model_path is None:
        if args.model == "v4-flash":
            raise ValueError("--model-path is required for v4-flash")
        model_path = "deepseek-ai/DeepSeek-V2-Lite"
    local_model = Path(model_path).expanduser()
    if local_model.exists() or model_path.startswith(("/", ".", "~")):
        model_path = str(local_model.resolve())
    if args.baseline:
        topology = args.topology or ("dp2tp4" if args.model == "v4-flash" else "dp4tp1")
    else:
        topology = args.topology or ("dp2tp4-ep8" if args.model == "v4-flash" else "2a2f")
    limit = FULL_GSM8K_COUNT if args.limit == "all" else int(args.limit)
    async_mode = args.comm == "async"
    v4 = args.model == "v4-flash"
    if v4 and async_mode:
        default_model_len, default_batched_tokens = 65536, 32768
    elif v4:
        default_model_len, default_batched_tokens = 8192, 1024
    elif async_mode:
        default_model_len, default_batched_tokens = 4096, 8000
    else:
        default_model_len, default_batched_tokens = 4096, 2048
    quant = args.dynamic_quant
    if quant is None:
        quant = int(v4 or (Path(model_path) / "quant_model_description.json").is_file())
    case = Case(
        model=args.model,
        model_path=model_path,
        repo=(args.repo or default_repo()).resolve(),
        topology=topology,
        comm=args.comm,
        mrv=args.mrv,
        ffn_mrv=args.ffn_mrv or args.mrv,
        execution=args.execution,
        dbo=args.dbo == "on",
        layered_gmm=args.layered_gmm == "on",
        moe_ubatching=(
            args.moe_ubatching == "on" if args.moe_ubatching else async_mode
        ),
        devices=devices,
        limit=limit,
        test=args.test if args.test is not None else ("perf" if args.baseline else "all"),
        performance=performance_configuration(args),
        dataset_path=args.dataset_path.resolve() if args.dataset_path else None,
        threshold=args.threshold,
        fewshot=args.fewshot,
        concurrency=args.concurrency,
        max_gen_tokens=args.max_gen_tokens,
        max_model_len=args.max_model_len
        if args.max_model_len is not None
        else default_model_len,
        max_batched_tokens=args.max_batched_tokens
        if args.max_batched_tokens is not None
        else default_batched_tokens,
        max_seqs=args.max_seqs if args.max_seqs is not None else 16,
        memory_utilization=args.memory_utilization,
        capture_sizes=tuple(int(v) for v in args.capture_sizes.split(",")),
        dynamic_quant=quant,
        api_host=args.api_host,
        api_port=args.api_port,
        afd_host=args.afd_host,
        afd_port=args.afd_port,
        nic=args.nic,
        python=args.python or sys.executable,
        vllm_bin=args.vllm_bin,
        offline=args.offline,
        startup_timeout=args.startup_timeout,
        eval_timeout=args.eval_timeout,
        request_timeout=args.request_timeout,
        cleanup_timeout=args.cleanup_timeout,
        run_dir=args.output.resolve() if args.output else None,
        baseline=args.baseline,
        common_args=tuple(args.vllm_arg),
        attention_args=tuple(args.attention_arg),
        ffn_args=tuple(args.ffn_arg),
        omit_args=tuple(args.omit_vllm_arg),
        serve_env=tuple(args.serve_env),
    )
    validate(case)
    return case


def validate(case: Case) -> None:
    tests = case.tests
    if (
        "perf" in tests
        and max(case.performance.input_lens) + max(case.performance.output_lens)
        > case.max_model_len
    ):
        raise ValueError("Performance input + output length exceeds --max-model-len")
    if not case.baseline and case.model == "v2-lite" and case.topology not in {"2a1f", "2a2f"}:
        raise ValueError("v2-lite supports 2a1f and 2a2f")
    if not case.baseline and case.model == "v4-flash" and case.topology != "dp2tp4-ep8":
        raise ValueError("v4-flash requires Attention DP2/TP4 + FFN DP8/TP1/EP8")
    if case.baseline and (
        case.comm != "sync"
        or case.layered_gmm
        or case.moe_ubatching
    ):
        raise ValueError(
            "Baseline uses native vLLM-Ascend without AFD communication or splitting"
        )
    if case.baseline:
        if "gsm8k" in tests:
            raise ValueError("Baseline supports perf and smoke; GSM8K routing is not enabled")
    if case.comm == "sync" and case.moe_ubatching:
        raise ValueError("--moe-ubatching configures the async deployment template")
    if case.devices is not None and len(case.devices) < case.required_devices:
        raise ValueError(
            f"Topology needs {case.required_devices} devices; "
            f"received {len(case.devices)}"
        )
    if not 1 <= case.limit <= FULL_GSM8K_COUNT:
        raise ValueError(f"--limit must be in 1..{FULL_GSM8K_COUNT}, or all")
    for name in (
        "concurrency",
        "max_gen_tokens",
        "max_model_len",
        "max_batched_tokens",
        "max_seqs",
        "startup_timeout",
        "eval_timeout",
        "request_timeout",
        "cleanup_timeout",
    ):
        if not math.isfinite(getattr(case, name)) or getattr(case, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if (
        case.fewshot < 0
        or not math.isfinite(case.threshold)
        or not 0 <= case.threshold <= 1
    ):
        raise ValueError(
            "fewshot must be nonnegative; threshold must be finite in [0, 1]"
        )
    if (
        not math.isfinite(case.memory_utilization)
        or not 0 < case.memory_utilization < 1
    ):
        raise ValueError("memory-utilization must be in (0, 1)")
    if not case.capture_sizes or any(size <= 0 for size in case.capture_sizes):
        raise ValueError("capture-sizes must contain positive integers")
    port_count = 2 * len(case.baseline_layouts) if case.baseline else 2
    if not 1 <= case.api_port <= 65536 - port_count or (not case.baseline and (
        not 1 <= case.afd_port <= 65535
        or case.afd_port in {case.api_port, case.api_port + 1}
    )):
        raise ValueError("API/AFD ports must be distinct and in range")
    reserved = {
        "--host",
        "--port",
        "--served-model-name",
        "--data-parallel-size",
        "--data-parallel-rpc-port",
        "--tensor-parallel-size",
        "--max-model-len",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--gpu-memory-utilization",
    }
    for token in (*case.common_args, *case.attention_args, *case.ffn_args):
        if token.split("=", 1)[0] in reserved:
            raise ValueError(
                f"{token} is owned by the harness; use its dedicated option"
            )
    for option in case.omit_args:
        if not option.startswith("--") or "=" in option or option in reserved:
            raise ValueError(f"Cannot omit harness-owned or invalid option: {option}")
    for assignment in case.serve_env:
        name, separator, _ = assignment.partition("=")
        if (
            not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
            or name in {"ASCEND_RT_VISIBLE_DEVICES", "AFD_E2E_RUN_ID",
                        "AFD_E2E_PROCESS_ROLE", "PYTHONPATH"}
        ):
            raise ValueError(f"Invalid or harness-owned serve environment: {assignment}")
