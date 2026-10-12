"""Device leases inside the current host/Pod; no cluster or Pod lifecycle."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

from .config import AllocatorConfig, Case, parse_devices


@dataclass(frozen=True)
class Lease:
    devices: tuple[str, ...]
    response: dict
    managed: bool


def invoke(command: tuple[str, ...], payload: dict, manager: AllocatorConfig) -> str:
    substitutions = {
        "count": str(payload.get("count", "")),
        "run_id": str(payload["run_id"]),
        "lease_id": str(payload.get("lease_id", "")),
    }
    tokens = list(command)
    for name, value in substitutions.items():
        tokens = [token.replace("{" + name + "}", value) for token in tokens]
    env = os.environ.copy()
    env["AFD_E2E_RUN_ID"] = payload["run_id"]
    env["AFD_E2E_REQUESTED_DEVICES"] = substitutions["count"]
    result = subprocess.run(
        tokens,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=manager.timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Device manager exited {result.returncode}: {result.stderr[-3000:]}"
        )
    return result.stdout


def release(case: Case, lease: Lease, run_id: str) -> None:
    if lease.managed:
        invoke(
            case.allocator.release,
            {**lease.response, "count": case.npu_count, "run_id": run_id},
            case.allocator,
        )


def acquire(case: Case, run_id: str) -> Lease:
    if not case.allocator.allocate:
        if case.devices is None:
            raise ValueError(
                "Specify --devices/--device, an inherited ASCEND_RT_VISIBLE_DEVICES, or a device allocator"
            )
        return Lease(case.devices, {}, False)
    raw = invoke(
        case.allocator.allocate,
        {"count": case.npu_count, "run_id": run_id},
        case.allocator,
    )
    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Device manager must print one JSON object to stdout; send logs to stderr"
        ) from exc
    if not isinstance(response, dict):
        raise RuntimeError("Allocation response must be a JSON object")
    # Once a parseable lease exists, release it even if its device list is bad.
    lease = Lease((), response, True)
    try:
        if not response.get("lease_id"):
            raise ValueError("Allocation response requires a nonempty lease_id")
        devices = parse_devices(response["devices"])
        if len(devices) != case.npu_count:
            raise ValueError(
                f"Requested {case.npu_count} devices, manager returned {len(devices)}"
            )
        inherited = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if inherited and not set(devices) <= set(parse_devices(inherited)):
            raise ValueError(
                "Allocated devices are outside inherited ASCEND_RT_VISIBLE_DEVICES"
            )
        return Lease(devices, response, True)
    except BaseException as original:
        try:
            release(case, lease, run_id)
        except Exception as cleanup:
            raise RuntimeError(
                f"Invalid allocation: {original}; release also failed: {cleanup}"
            ) from original
        raise
