# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# SPDX-License-Identifier: Apache-2.0
"""Build runtime commands without importing torch, vLLM, or repository tests."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import struct
from dataclasses import dataclass
from pathlib import Path

from .config import TOOL_ROOT, Case
from .routing import routes


@dataclass(frozen=True)
class Service:
    role: str
    devices: tuple[str, ...]
    command: tuple[str, ...]
    environment: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "devices": self.devices,
            "command": self.command,
            "environment": self.environment,
        }


def network(case: Case) -> tuple[str, str | None]:
    interface = case.nic or os.environ.get("HCCL_SOCKET_IFNAME")
    address = case.afd_host or os.environ.get("HCCL_IF_IP")
    if interface is None:
        try:
            for line in Path("/proc/net/route").read_text().splitlines()[1:]:
                values = line.split()
                if values[1] == "00000000" and int(values[3], 16) & 2:
                    interface = values[0]
                    break
        except (OSError, IndexError, ValueError):
            pass
    if address is None and interface:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                value = fcntl.ioctl(
                    sock.fileno(), 0x8915, struct.pack("256s", interface.encode()[:15])
                )
            address = socket.inet_ntoa(value[20:24])
        except OSError:
            pass
    return address or "127.0.0.1", interface


def environment(
    case: Case, role: str, devices: tuple[str, ...], run_id: str
) -> dict[str, str]:
    address, interface = network(case)
    env = {
        "PYTHONUNBUFFERED": "1",
        "VLLM_USE_V1": "1",
        "VLLM_PLUGINS": (
            "ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
            if case.baseline else "ascend,afd"
        ),
        "VLLM_USE_V2_MODEL_RUNNER": str(
            int((case.ffn_mrv if role == "ffn" else case.mrv) == "mrv2")
        ),
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(devices),
        "AFD_E2E_RUN_ID": run_id,
        "AFD_E2E_PROCESS_ROLE": role,
        "AFD_ASYNC_CAM_LAYERED_GMM": str(int(case.layered_gmm and role == "ffn")),
        "PYTHONPATH": os.pathsep.join(
            filter(None, (
                str(case.repo) if not case.baseline else None,
                str(TOOL_ROOT), os.environ.get("PYTHONPATH"),
            ))
        ),
        "PYTORCH_NPU_ALLOC_CONF": os.environ.get(
            "PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True"
        ),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
        "HCCL_IF_IP": address,
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_EXEC_TIMEOUT": "2000",
        "VLLM_RPC_TIMEOUT": "3600000",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
        "VLLM_ENGINE_READY_TIMEOUT_S": str(int(case.startup_timeout)),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8" if case.baseline else "10"),
        "OMP_PROC_BIND": "false",
        "AFD_FORCE_BALANCED_TOPK_IDS": "0",
        "AFD_CAM_OP_IO_LOG": "0",
        "AFD_ASYNC_MOE_LAYOUT_LOG": "0",
    }
    if interface:
        env.update(
            {
                name: interface
                for name in (
                    "HCCL_SOCKET_IFNAME",
                    "GLOO_SOCKET_IFNAME",
                    "TP_SOCKET_IFNAME",
                )
            }
        )
    if case.comm == "async":
        env.update(
            {
                "HCCL_BUFFSIZE": "4096",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "TASK_QUEUE_ENABLE": "1",
            }
        )
    if case.offline:
        env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    env.update(assignment.split("=", 1) for assignment in case.serve_env)
    return env


def runtime_environment(service: Service) -> dict[str, str]:
    env = os.environ.copy()
    env.update(service.environment)
    return env


def _additional_config(case: Case, role: str, address: str) -> dict:
    config = {}
    if not case.baseline:
        extra = {"hccl_buffer_size": 4096 if case.comm == "async" else 2048}
        if case.comm == "async":
            extra.update(
                {
                    "dynamicQuant": case.dynamic_quant,
                    "attn_ranks_per_dp": case.attention_tp,
                    "async_moe_ubatching": case.moe_ubatching,
                }
            )
        else:
            extra["quant_mode"] = 0
        config["afd"] = {
            "role": role,
            "connector": "CAMAsyncAFDConnector"
            if case.comm == "async"
            else "CAMP2pAFDConnector",
            "host": address,
            "port": case.afd_port,
            "num_attention_ranks": case.attention_ranks,
            "num_ffn_ranks": case.ffn_ranks,
            "async": case.comm == "async",
            "compute_gate_on_attention": case.comm == "async",
            "connector_extra_config": extra,
        }
    if case.model == "v4-flash":
        config.update(
            {
                "enable_flashcomm1": case.comm == "async" and role != "ffn",
                "enable_cpu_binding": True,
                "enable_force_eplb": False,
                "enable_dsa_cp": False,
                "multistream_dsv4_dsa_overlap": False,
            }
        )
    return config


def _service_command(
    case: Case, role: str, address: str, count: int, tp: int, instance: int = 0
) -> tuple[str, ...]:
    ffn = role == "ffn"
    config = _additional_config(case, role, address)
    command = [
        case.vllm_bin,
        "serve",
        case.model_path,
        "--served-model-name",
        f"afd-e2e-{role}",
        "--host",
        case.api_host,
        "--port",
        str(case.api_port + (instance if case.baseline else int(ffn))),
        "--shutdown-timeout",
        "10",
        "--data-parallel-size",
        str(count // tp),
        "--tensor-parallel-size",
        str(tp),
        "--enable-expert-parallel",
        "--max-model-len",
        str(case.max_model_len),
        "--max-num-batched-tokens",
        str(case.max_batched_tokens),
        "--max-num-seqs",
        str(case.max_seqs),
        "--gpu-memory-utilization",
        str(case.memory_utilization),
        "--seed",
        "42",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--enable-chunked-prefill",
    ]
    if case.baseline:
        command.extend((
            "--data-parallel-rpc-port",
            str(case.api_port + len(case.baseline_layouts) + instance),
        ))
    if config:
        command.extend(
            ("--additional-config", json.dumps(config, separators=(",", ":")))
        )
    if case.execution == "eager":
        command.append("--enforce-eager")
    else:
        command.extend(
            (
                "--compilation-config",
                json.dumps(
                    {
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": case.capture_sizes,
                    },
                    separators=(",", ":"),
                ),
            )
        )
    native_dbo = case.dbo or (case.comm == "async" and case.moe_ubatching)
    if native_dbo and (case.comm != "async" or role == "attention"):
        if case.dbo:
            command.append("--enable-dbo")
        command.extend(
            (
                "--dbo-decode-token-threshold",
                "2",
                "--dbo-prefill-token-threshold",
                "8",
            )
        )
    if case.model == "v4-flash":
        command.extend(
            (
                "--block-size",
                "128",
                "--quantization",
                "ascend",
                "--attention_config.indexer_kv_dtype",
                "int8",
                "--tokenizer-mode",
                "deepseek_v4",
                "--model-loader-extra-config",
                '{"enable_multithread_load":true,"num_threads":128}',
            )
        )
        if not ffn:
            command.extend(
                (
                    "--data-parallel-address",
                    address,
                    "--tool-call-parser",
                    "deepseek_v4",
                    "--enable-auto-tool-choice",
                    "--reasoning-parser",
                    "deepseek_v4",
                    "--no-disable-hybrid-kv-cache-manager",
                )
            )
    if case.comm == "async":
        command.extend(
            (
                "--all2all-backend",
                "flashinfer_all2allv" if ffn else "allgather_reducescatter",
            )
        )
    for option in case.omit_args:
        command = _without_option(command, option)
    for arguments in (case.common_args, case.ffn_args if ffn else case.attention_args):
        for token in arguments:
            if token.startswith("--"):
                option = token.split("=", 1)[0]
                command = _without_option(command, option)
                opposite = (
                    "--" + option[5:] if option.startswith("--no-")
                    else "--no-" + option[2:]
                )
                command = _without_option(command, opposite)
        command.extend(arguments)
    return tuple(command)


def _without_option(command: list[str], option: str) -> list[str]:
    result = []
    index = 0
    while index < len(command):
        token = command[index]
        if token.split("=", 1)[0] == option:
            index += 1
            while index < len(command) and not command[index].startswith("--"):
                index += 1
        else:
            result.append(token)
            index += 1
    return result


def build_services(
    case: Case, devices: tuple[str, ...], run_id: str
) -> tuple[Service, ...]:
    if len(devices) < case.required_devices:
        raise ValueError("Not enough assigned devices")
    address, _ = network(case)
    if case.baseline:
        services = []
        offset = 0
        for instance, (dp, tp) in enumerate(case.baseline_layouts):
            assigned = devices[offset : offset + dp * tp]
            role = f"baseline-{instance}"
            services.append(Service(
                role, assigned,
                _service_command(case, "baseline", address, dp * tp, tp, instance),
                environment(case, role, assigned, run_id),
            ))
            offset += dp * tp
        return tuple(services)
    if case.comm == "async":
        roles = ("attention", "ffn")
    else:
        roles = ("ffn", "attention")
    services = []
    for role in roles:
        ffn = role == "ffn"
        count = case.ffn_ranks if ffn else case.attention_ranks
        assigned = (
            devices[case.attention_ranks : case.required_devices]
            if ffn
            else devices[: case.attention_ranks]
        )
        tp = 1 if ffn else case.attention_tp
        services.append(
            Service(
                role,
                assigned,
                _service_command(case, role, address, count, tp),
                environment(case, role, assigned, run_id),
            )
        )
    return tuple(services)


def describe(case: Case) -> dict:
    from .performance import points

    devices = case.devices or tuple(str(i) for i in range(case.required_devices))
    result = {
        "configuration": case.to_dict(),
        "device_assignment_pending": case.devices is None,
        "routes": routes(case.to_dict()),
        "services": [
            service.to_dict() for service in build_services(case, devices, "PLAN-ONLY")
        ],
        "unused_devices": devices[case.required_devices :],
        "performance_points": points(case.to_dict()["performance"])
        if "perf" in case.tests
        else [],
        "notes": [
            "plan does not allocate devices or start services",
        ],
    }
    if case.baseline:
        result["native_instances"] = [
            {"instance": i, "dp": dp, "tp": tp, "ep": dp * tp}
            for i, (dp, tp) in enumerate(case.baseline_layouts)
        ]
    else:
        result.update(
            attention={"ranks": case.attention_ranks,
                       "dp": case.attention_ranks // case.attention_tp,
                       "tp": case.attention_tp},
            ffn={"ranks": case.ffn_ranks, "dp": case.ffn_ranks,
                 "tp": 1, "ep": case.ffn_ranks},
        )
    result["notes"].append(
        "Tests use HTTP responses, client timings, service logs and optional /metrics; "
        "internal runner, graph, splitting and kernel execution are not verified"
    )
    return result
