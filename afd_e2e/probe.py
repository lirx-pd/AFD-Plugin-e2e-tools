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
    check_ports(args.api_host, tuple(args.ports))
    if args.nic:
        socket.if_nametoindex(args.nic)
    model = Path(args.model)
    hashes = {}
    config = {}
    shards = set()
    if model.is_dir():
        config = json.loads((model / "config.json").read_text())
        indexes = tuple(model.glob("*.safetensors.index.json"))
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
            if (
                not path.is_relative_to(model.resolve())
                or not path.is_file()
                or path.stat().st_size == 0
            ):
                raise RuntimeError(f"Invalid checkpoint shard: {name}")
        for path in (
            model / "config.json",
            model / "tokenizer_config.json",
            model / "quant_model_description.json",
            *indexes,
        ):
            if path.is_file():
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
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
        "model_type": config.get("model_type"),
        "layers": config.get("num_hidden_layers"),
        "weight_shards": len(shards),
        "model_hashes": hashes,
        "requested_devices": args.devices.split(","),
        "hostname": socket.gethostname(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--api-host", required=True)
    parser.add_argument(
        "--ports", required=True, type=lambda value: [int(p) for p in value.split(",")]
    )
    parser.add_argument("--nic")
    args = parser.parse_args()
    print("AFD_E2E_PROBE " + json.dumps(probe(args), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
