import asyncio
import builtins
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from afd_e2e import performance, probe, runner, workloads
from afd_e2e.planner import build_services, describe
from afd_e2e.routing import routes


@pytest.mark.parametrize(
    "model,topology,layouts,total",
    [
        ("v2-lite", "dp4", [(4, 1)], 4),
        ("v4-flash", "dp2", [(2, 4)], 8),
        ("v4-flash", "dp2,dp2", [(2, 4), (2, 4)], 16),
        ("v4-flash", "dp1,dp1,dp1,dp1", [(1, 4)] * 4, 16),
        ("v2-lite", "dp1tp2,dp3", [(1, 2), (3, 1)], 5),
    ],
)
def test_native_dp_instances_and_domains(make_case, model, topology, layouts, total):
    case = make_case(
        "--baseline", "--model", model, "--model-path", "/weights",
        "--topology", topology, "--repo", "/missing/afd-plugin",
    )
    assert case.test == "perf" and case.required_devices == total
    devices = tuple(str(i + 20) for i in range(total))
    services = build_services(case, devices, "test")
    plan = describe(case)
    assert len(services) == len(layouts)
    assert tuple(d for service in services for d in service.devices) == devices
    assert len(plan["routes"]) == sum(dp for dp, _ in layouts)
    ports = []
    for i, (service, (dp, tp)) in enumerate(zip(services, layouts)):
        args = service.command
        assert args[args.index("--data-parallel-size") + 1] == str(dp)
        assert args[args.index("--tensor-parallel-size") + 1] == str(tp)
        assert len(service.devices) == dp * tp
        assert "afd" not in service.environment["VLLM_PLUGINS"].split(",")
        assert "ascend_model" in service.environment["VLLM_PLUGINS"].split(",")
        assert "/missing/afd-plugin" not in service.environment["PYTHONPATH"]
        assert "--worker-extension-cls" not in args
        if "--additional-config" in args:
            assert "afd" not in json.loads(args[args.index("--additional-config") + 1])
        ports.extend(int(args[args.index(flag) + 1]) for flag in (
            "--port", "--data-parallel-rpc-port"
        ))
        assert service.role == f"baseline-{i}"
    assert len(ports) == len(set(ports))
    assert set(ports) == set(range(case.api_port, case.api_port + 2 * len(layouts)))


@pytest.mark.parametrize("topology", ["dp0", "dp2tp0", "ep8", "dp2,", "2a2f"])
def test_invalid_native_topology(make_case, topology):
    with pytest.raises(ValueError, match="Baseline topology"):
        make_case("--baseline", "--topology", topology)


def test_baseline_device_and_port_boundaries(make_case):
    with pytest.raises(ValueError, match="Topology needs 5 devices"):
        make_case("--baseline", "--topology", "dp2,dp3", "--device", "0,1,2,3")
    with pytest.raises(ValueError, match="ports"):
        make_case("--baseline", "--topology", "dp1,dp1", "--api-port", "65533")
    case = make_case("--baseline", "--topology", "dp1,dp1", "--api-port", "65532")
    assert case.required_devices == 2


def test_native_preflight_does_not_import_afd(monkeypatch, tmp_path):
    check_ports = MagicMock()
    monkeypatch.setattr(probe, "check_ports", check_ports)
    torch = MagicMock()
    torch.npu.device_count.return_value = 2
    torch.testing.assert_close = MagicMock()
    for name, module in {
        "torch": torch, "torch_npu": SimpleNamespace(),
        "vllm": SimpleNamespace(__file__="/native/vllm/__init__.py"),
        "vllm_ascend": SimpleNamespace(),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    original_import = builtins.__import__

    def native_import(name, *args, **kwargs):
        assert not name.startswith("afd_plugin")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", native_import)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"deepseek_v2"}')
    (model / "tokenizer_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture")
    args = SimpleNamespace(
        ports=[19280, 19281, 19282, 19283], api_host="127.0.0.1",
        api_port=19280, afd_port=6455, nic=None, repo="/missing",
        baseline=True, devices="0,1", model=str(model), family="v2-lite",
        async_cam=False, layered=False,
    )
    assert probe.probe(args)["afd_source"] is None
    check_ports.assert_called_once_with("127.0.0.1", tuple(args.ports))


def test_all_native_instances_must_be_ready(make_case, monkeypatch):
    case = make_case("--baseline", "--topology", "dp1,dp2", "--startup-timeout", "0.01")
    visited = []

    class Opener:
        def open(self, url, **_):
            visited.append(url)
            if url.endswith(f":{case.api_port + 1}/v1/models"):
                raise OSError("second instance is not ready")
            stream = io.StringIO('{"data":[{"id":"afd-e2e-baseline"}]}')
            stream.status = 200
            return stream

    monkeypatch.setattr(runner.urllib.request, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match=str(case.api_port + 1)):
        runner.wait_ready(case, [])
    assert f"http://127.0.0.1:{case.api_port}/v1/models" in visited
    assert f"http://127.0.0.1:{case.api_port + 1}/v1/models" in visited


def test_native_perf_and_smoke_send_url_and_dp_rank(make_case, monkeypatch, tmp_path):
    case = make_case(
        "--baseline", "--topology", "dp1,dp2", "--concurrency", "6",
        "--perf-input-lens", "2", "--perf-output-lens", "1",
        "--perf-concurrencies", "3", "--perf-mode", "matrix",
    )
    endpoints = routes(case.to_dict())
    seen = []

    async def handler(request):
        body = json.loads(request.content)
        index = len(seen)
        endpoint = endpoints[index % len(endpoints)]
        assert str(request.url).startswith(endpoint["url"] + "/v1/")
        assert request.headers["X-data-parallel-rank"] == str(endpoint["dp_rank"])
        seen.append((str(request.url), request.headers["X-data-parallel-rank"]))
        await asyncio.sleep(0.01)
        if body.get("stream"):
            values = [
                {"choices": [{"text": "x", "token_ids": [1], "finish_reason": "length"}]},
                {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
            ]
            content = "".join("data: " + json.dumps(v) + "\n\n" for v in values)
            return httpx.Response(200, content=content + "data: [DONE]\n\n")
        return httpx.Response(200, json={
            "choices": [{"text": "19", "finish_reason": "stop"}],
        })

    client_class = httpx.AsyncClient

    async def perf_requests():
        async with client_class(transport=httpx.MockTransport(handler)) as client:
            records = []
            await performance.batch(
                client, endpoints, "afd-e2e-baseline", [1, 2],
                performance.points(case.to_dict()["performance"])[0],
                6, "measured", 5, records, io.StringIO(), {},
            )
            assert all(r["status"] == "passed" for r in records)
            assert [r["route"]["domain"] for r in records] == [0, 1, 2, 0, 1, 2]

    asyncio.run(perf_requests())
    assert len(seen) == 6
    seen.clear()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handler), **kwargs
    ))
    output = tmp_path / "smoke.json"
    workloads.evaluate(case.to_dict(), "smoke", output)
    assert len(seen) == 6
    assert [r["route"]["domain"] for r in json.loads(output.read_text())["requests"]] == [0, 1, 2, 0, 1, 2]
