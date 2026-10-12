# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from afd_e2e import cli, config, evaluation
from afd_e2e.devices import acquire, release
from afd_e2e.planner import build_services, describe, runtime_environment


@pytest.mark.parametrize(
    "comm,topology,dp,tp,total",
    [
        ("sync", "2a1f", 2, 1, 3),
        ("sync", "2a2f", 2, 1, 4),
        ("async", "2a1f", 1, 2, 3),
        ("async", "2a2f", 1, 2, 4),
    ],
)
def test_v2_layouts(make_case, comm, topology, dp, tp, total):
    case = make_case(
        "--comm",
        comm,
        "--topology",
        topology,
        "--device",
        ",".join(map(str, range(total))),
    )
    plan = describe(case)
    assert plan["attention"] == {"ranks": 2, "dp": dp, "tp": tp}
    assert case.required_devices == total
    services = {s.role: s for s in build_services(case, case.devices, "test")}
    assert services["attention"].devices == ("0", "1")
    assert services["ffn"].devices == tuple(map(str, range(2, total)))
    connector = json.loads(
        services["attention"].command[
            services["attention"].command.index("--additional-config") + 1
        ]
    )["afd"]
    assert connector["async"] == (comm == "async")
    assert connector["num_ffn_ranks"] == total - 2


@pytest.mark.parametrize("ffn_mrv", ["mrv1", "mrv2"])
def test_v4_layered_layout_and_flags(make_case, ffn_mrv):
    case = make_case(
        "--model",
        "v4-flash",
        "--model-path",
        "/weights/v4",
        "--comm",
        "async",
        "--mrv",
        "mrv2",
        "--ffn-mrv",
        ffn_mrv,
        "--layered-gmm",
        "on",
    )
    plan = describe(case)
    assert case.required_devices == 16
    assert plan["attention"] == {"ranks": 8, "dp": 2, "tp": 4}
    assert plan["ffn"] == {"ranks": 8, "dp": 8, "tp": 1, "ep": 8}
    attention, ffn = build_services(case, tuple(map(str, range(16))), "test")
    assert attention.environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert ffn.environment["VLLM_USE_V2_MODEL_RUNNER"] == str(int(ffn_mrv == "mrv2"))
    assert ffn.environment["AFD_ASYNC_CAM_LAYERED_GMM"] == "1"
    assert attention.devices == tuple(map(str, range(8)))
    assert ffn.devices == tuple(map(str, range(8, 16)))


@pytest.mark.parametrize(
    "arguments",
    [
        ["--comm", "async", "--execution", "graph"],
        ["--mrv", "mrv2", "--dbo", "on"],
        ["--mrv", "mrv2", "--ffn-mrv", "mrv1", "--dbo", "on", "--execution", "graph"],
        ["--model", "v4-flash"],
        [
            "--model",
            "v4-flash",
            "--model-path",
            "/w",
            "--comm",
            "async",
            "--dynamic-quant",
            "0",
            "--layered-gmm",
            "on",
        ],
        ["--moe-ubatching", "on"],
        ["--npu-count", "0"],
        ["--limit", "0"],
        ["--limit", "1320"],
        ["--threshold", "nan"],
        ["--max-model-len", "0"],
        ["--eval-timeout", "inf"],
        ["--devices", "0,1,1,2"],
        ["--execution", "graph", "--max-seqs", "32"],
        ["--vllm-arg=--enable-dbo"],
    ],
)
def test_rejects_invalid_configuration(make_case, arguments):
    with pytest.raises(ValueError):
        make_case(*arguments)


def test_exact_gsm8k_limit_with_dbo(make_case, tmp_path):
    case = make_case("--dbo", "on", "--limit", "7")
    command = evaluation.command(case, tmp_path)
    assert command[command.index("--limit") + 1] == "7"
    assert make_case("--limit", "all").limit == 1319


def test_native_mrv2_dbo_keeps_ffn_mrv1(make_case):
    case = make_case("--mrv", "mrv2", "--ffn-mrv", "mrv1", "--dbo", "on")
    devices = tuple(map(str, range(case.required_devices)))
    services = {
        service.role: service for service in build_services(case, devices, "p2a")
    }
    attention, ffn = services["attention"], services["ffn"]
    assert attention.environment["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert ffn.environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert "--enable-dbo" in attention.command
    assert "--enable-dbo" in ffn.command


def test_platform_is_optional_and_sibling_found_from_any_cwd(
    make_case, monkeypatch, tmp_path
):
    original = config.default_repo()
    monkeypatch.chdir(tmp_path)
    assert make_case("--device", "4,5,6,7").repo == original
    assert acquire(make_case("--device", "4,5,6,7"), "test").devices == (
        "4",
        "5",
        "6",
        "7",
    )


def test_inherited_environment_preserved(make_case, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("PYTHONPATH", "/cann/python")
    monkeypatch.setenv("AFD_CUST_OPAPI_LIB_PATH", "/custom/lib.so")
    case = make_case()
    assert case.devices == ("4", "5", "6", "7")
    env = runtime_environment(build_services(case, case.devices, "test")[0])
    assert "/cann/python" in env["PYTHONPATH"].split(":")
    assert env["AFD_CUST_OPAPI_LIB_PATH"] == "/custom/lib.so"
    with pytest.raises(ValueError, match="outside inherited"):
        make_case("--device", "0,1,2,3")


def test_service_registers_custom_opp_before_can_initialization(
    make_case, monkeypatch, tmp_path
):
    vendor = tmp_path / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    vendor.mkdir(parents=True)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/other/vendor")
    case = make_case("--repo", str(tmp_path))
    services = build_services(case, ("0", "1", "2", "3"), "test")
    for service in services:
        assert runtime_environment(service)["ASCEND_CUSTOM_OPP_PATH"].split(":") == [
            str(vendor),
            "/other/vendor",
        ]


def test_plan_never_calls_manager(manager, monkeypatch, capsys):
    platform, record = manager()
    monkeypatch.setattr(
        "afd_e2e.devices.invoke", lambda *args: pytest.fail("allocated during plan")
    )
    assert cli.main(["plan", "--platform", str(platform)]) == 0
    assert json.loads(capsys.readouterr().out)["device_assignment_pending"]
    assert not record.exists()


def test_manager_contract_and_release(make_case, manager):
    platform, record = manager()
    case = make_case("--platform", platform)
    lease = acquire(case, "run-safe")
    assert lease.devices == ("0", "1", "2", "3")
    release(case, lease, "run-safe")
    assert json.loads(record.read_text()) == {
        "devices": [0, 1, 2, 3],
        "lease_id": "lease-123",
        "count": 4,
        "run_id": "run-safe",
    }


@pytest.mark.parametrize("mode", ["duplicate", "short", "wrong_type"])
def test_invalid_manager_devices_release_lease(make_case, manager, mode):
    platform, record = manager(mode)
    with pytest.raises(ValueError):
        acquire(make_case("--platform", platform), "bad-allocation")
    assert json.loads(record.read_text())["lease_id"] == "lease-123"


def test_manager_cannot_expand_inherited_devices(make_case, manager, monkeypatch):
    platform, record = manager()
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    with pytest.raises(ValueError, match="outside inherited"):
        acquire(make_case("--platform", platform), "bad-allocation")
    assert record.exists()


def test_bad_platform_is_cli_error(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("allocator: [unclosed")
    assert cli.main(["plan", "--platform", str(path)]) == 2
    assert "Invalid YAML" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, 3, {}, [True], [1, 1], "1,,2", "-1,2", "１,2"])
def test_device_response_boundary(value):
    with pytest.raises(ValueError):
        config.parse_devices(value)


def test_yaml_and_json_platform_equivalent(tmp_path, make_case):
    yaml = tmp_path / "p.yaml"
    yaml.write_text(
        "allocator:\n  allocate: 'allocator --count {count}'\n  release: 'allocator --release {lease_id}'\n"
    )
    case = make_case("--platform", yaml)
    assert case.allocator.allocate == ("allocator", "--count", "{count}")
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_case("--platform", yaml, "--device", "0,1,2,3")


def test_async_dbo_is_enabled_only_on_attention(make_case):
    case = make_case(
        "--model",
        "v4-flash",
        "--model-path",
        "/w",
        "--comm",
        "async",
        "--mrv",
        "mrv2",
        "--ffn-mrv",
        "mrv1",
        "--dbo",
        "on",
    )
    attention, ffn = build_services(case, tuple(map(str, range(16))), "test")
    assert "--enable-dbo" in attention.command
    assert "--enable-dbo" not in ffn.command
