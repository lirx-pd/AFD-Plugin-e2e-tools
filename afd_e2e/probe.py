"""Executed in the target Python environment, never imported by plan."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import socket
from pathlib import Path


def check_ports(host: str, ports: tuple[int, ...]) -> None:
    for port in ports:
        with socket.socket() as endpoint:
            # Match server binding so a previous run's TIME_WAIT is reusable.
            endpoint.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            endpoint.bind((host, port))


def probe(args) -> dict:
    check_ports(args.api_host, (args.api_port, args.api_port + 1, args.afd_port))
    if args.nic:
        socket.if_nametoindex(args.nic)
    import torch
    import torch_npu  # noqa: F401
    import vllm
    import vllm_ascend  # noqa: F401
    import afd_plugin

    repo = Path(args.repo).resolve()
    if not Path(afd_plugin.__file__).resolve().is_relative_to(repo):
        raise RuntimeError(
            f"Wrong afd_plugin source: {afd_plugin.__file__}; expected {repo}"
        )
    plugins = importlib.metadata.entry_points(group="vllm.general_plugins")
    if not args.baseline and not any(entry.name == "afd" for entry in plugins):
        raise RuntimeError(
            "afd-plugin entry point missing; install/build the sibling afd-plugin in the target environment"
        )
    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable")
    requested = len(args.devices.split(","))
    if torch.npu.device_count() < requested:
        raise RuntimeError(
            f"Need {requested} visible devices, found {torch.npu.device_count()}"
        )
    for device in range(requested):
        reference = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        actual = reference.to(f"npu:{device}")
        torch.testing.assert_close((actual @ actual.T).cpu(), reference @ reference.T)
    model = Path(args.model)
    hashes = {}
    if model.is_dir():
        config = json.loads((model / "config.json").read_text())
        if not (model / "tokenizer_config.json").is_file():
            raise RuntimeError("Missing tokenizer_config.json")
        indexes = tuple(model.glob("*.safetensors.index.json"))
        shards = set()
        for index in indexes:
            shards.update(json.loads(index.read_text())["weight_map"].values())
        if not shards:
            shards = {path.name for path in model.glob("*.safetensors")}
        if not shards:
            shards = {path.name for path in model.glob("pytorch_model*.bin")}
        if not shards:
            raise RuntimeError("No checkpoint weights found")
        for name in shards:
            path = (model / name).resolve(strict=True)
            if not path.is_relative_to(model.resolve()) or path.stat().st_size == 0:
                raise RuntimeError(f"Invalid checkpoint shard: {name}")
        for path in (
            model / "config.json",
            model / "tokenizer_config.json",
            model / "quant_model_description.json",
            *indexes,
        ):
            if path.is_file():
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            args.model, trust_remote_code=True
        ).to_dict()
        shards = set()
    expected = "deepseek_v4" if args.family == "v4-flash" else "deepseek_v2"
    if config["model_type"] != expected:
        raise RuntimeError(f"Expected {expected}, found {config['model_type']}")
    if args.async_cam:
        from afd_plugin.compat.npu.ops import ensure_cam_async_ops_available

        ensure_cam_async_ops_available()
    elif not args.baseline:
        from afd_plugin.compat.npu.ops import ensure_cam_p2p_ops_available

        ensure_cam_p2p_ops_available()
    if args.layered:
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() != AscendDeviceType.A3:
            raise RuntimeError("Layered W4A8 GMM requires Ascend A3/910C")
        for name in ("grouped_matmul_layered", "gmm_swiglu_quant_v2_layered"):
            getattr(torch.ops.afd_ascend, name)
    versions = {}
    for name in (
        "torch",
        "torch-npu",
        "vllm",
        "vllm-ascend",
        "vllm-afd-plugin",
        "lm_eval",
        "datasets",
        "transformers",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "vllm_source": vllm.__file__,
        "afd_source": afd_plugin.__file__,
        "model_type": config["model_type"],
        "layers": config.get("num_hidden_layers"),
        "weight_shards": len(shards),
        "model_hashes": hashes,
        "tested_devices": args.devices.split(","),
        "hostname": socket.gethostname(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--api-host", required=True)
    parser.add_argument("--api-port", type=int, required=True)
    parser.add_argument("--afd-port", type=int, required=True)
    parser.add_argument("--nic")
    parser.add_argument("--async-cam", action="store_true")
    parser.add_argument("--layered", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    print("AFD_E2E_PROBE " + json.dumps(probe(args), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
