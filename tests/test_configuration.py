# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from afd_e2e import cli, config, evaluation, runner
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
        ["--model", "v4-flash"],
        ["--moe-ubatching", "on"],
        ["--vllm-arg=--port=10000"],
        ["--omit-vllm-arg=--data-parallel-size"],
        ["--omit-vllm-arg=shutdown-timeout"],
        ["--limit", "0"],
        ["--limit", "1320"],
        ["--threshold", "nan"],
        ["--max-model-len", "0"],
        ["--eval-timeout", "inf"],
        ["--devices", "0,1,1,2"],
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


def test_sibling_found_from_any_cwd(make_case, monkeypatch, tmp_path):
    original = config.default_repo()
    monkeypatch.chdir(tmp_path)
    case = make_case("--device", "4,5,6,7")
    assert case.repo == original
    assert case.devices == ("4", "5", "6", "7")


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


def test_service_inherits_custom_opp_without_guessing_plugin_paths(
    make_case, monkeypatch, tmp_path
):
    vendor = tmp_path / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    vendor.mkdir(parents=True)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/other/vendor")
    case = make_case("--repo", str(tmp_path))
    services = build_services(case, ("0", "1", "2", "3"), "test")
    for service in services:
        assert runtime_environment(service)["ASCEND_CUSTOM_OPP_PATH"] == "/other/vendor"


def test_plan_without_devices_keeps_assignment_pending(monkeypatch, capsys):
    monkeypatch.setattr(
        runner, "ManagedProcess", lambda *args: pytest.fail("started during plan")
    )
    assert cli.main(["plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["device_assignment_pending"]
    assert plan["configuration"]["devices"] is None
    assert sum(len(service["devices"]) for service in plan["services"]) == 4


@pytest.mark.parametrize(
    "option,value",
    [
        ("--platform", "platform.json"),
        ("--allocate-command", "allocate"),
        ("--release-command", "release"),
        ("--npu-count", "4"),
    ],
)
def test_removed_device_options_are_cli_errors(option, value):
    with pytest.raises(SystemExit) as error:
        cli.main(["plan", option, value])
    assert error.value.code == 2


def test_explicit_devices_can_select_inherited_subset(make_case, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7,8,9")
    case = make_case("--device", "6,4,9,5")
    assert case.devices == ("6", "4", "9", "5")
    services = {s.role: s for s in build_services(case, case.devices, "test")}
    assert services["attention"].devices == ("6", "4")
    assert services["ffn"].devices == ("9", "5")


@pytest.mark.parametrize("topology,devices", [("2a1f", "0,1"), ("2a2f", "0,1,2")])
def test_devices_must_cover_topology(make_case, topology, devices):
    with pytest.raises(ValueError, match="Topology needs"):
        make_case("--topology", topology, "--device", devices)


def test_extra_devices_are_unused(make_case):
    case = make_case("--device", "4,5,6,7,8")
    plan = describe(case)
    assert plan["unused_devices"] == ("8",)
    services = {s.role: s for s in build_services(case, case.devices, "test")}
    assert services["attention"].devices == ("4", "5")
    assert services["ffn"].devices == ("6", "7")


def test_run_without_devices_fails_before_creating_output(make_case, tmp_path):
    case = make_case("--repo", tmp_path, "--output", tmp_path / "run")
    with pytest.raises(ValueError, match=r"--device.*npu run|npu run.*--device"):
        runner.execute(case)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [None, 3, {}, [True], [1, 1], "1,,2", "-1,2", "１,2"])
def test_device_response_boundary(value):
    with pytest.raises(ValueError):
        config.parse_devices(value)


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


@pytest.mark.parametrize(
    "flags",
    [
        ["--comm", "async", "--execution", "graph"],
        ["--mrv", "mrv2", "--dbo", "on", "--execution", "graph"],
        ["--model", "v4-flash", "--model-path", "/w", "--layered-gmm", "on"],
        ["--execution", "graph", "--max-seqs", "32"],
        ["--comm", "async", "--max-batched-tokens", "17"],
    ],
)
def test_runtime_decides_feature_support(make_case, flags):
    case = make_case(*flags)
    services = build_services(case, tuple(map(str, range(case.required_devices))), "test")
    assert services
    assert all("--worker-extension-cls" not in service.command for service in services)


def test_serve_arguments_override_presets_and_common_arguments(make_case):
    case = make_case(
        "--execution", "graph", "--dbo", "on",
        "--vllm-arg=--compilation-config", '--vllm-arg={"mode":0}',
        "--vllm-arg=--no-enable-dbo", "--vllm-arg=--enable-prefix-caching",
        '--vllm-arg=--additional-config={"common":true}',
        "--attention-arg=--additional-config", '--attention-arg={"new_afd":true}',
        "--vllm-arg=--lora-modules", "--vllm-arg=a=/a", "--vllm-arg=b=/b",
        "--attention-arg=--lora-modules=c=/c",
    )
    services = {s.role: s for s in build_services(case, ("0", "1", "2", "3"), "test")}
    attention, ffn = services["attention"].command, services["ffn"].command
    for command in (attention, ffn):
        assert command.count("--compilation-config") == 1
        assert command[command.index("--compilation-config") + 1] == '{"mode":0}'
        assert "--enable-dbo" not in command
        assert "--no-enable-dbo" in command
        assert "--no-enable-prefix-caching" not in command
        assert "--enable-prefix-caching" in command
    assert attention[attention.index("--additional-config") + 1] == '{"new_afd":true}'
    assert '--additional-config={"common":true}' in ffn
    assert "a=/a" not in attention and "b=/b" not in attention
    assert "--lora-modules=c=/c" in attention
    assert "a=/a" in ffn and "b=/b" in ffn


def test_omit_version_specific_serve_options(make_case):
    case = make_case(
        "--execution", "graph",
        "--omit-vllm-arg=--shutdown-timeout",
        "--omit-vllm-arg=--compilation-config",
        "--omit-vllm-arg=--enable-chunked-prefill",
        "--vllm-arg=--enforce-eager",
    )
    for service in build_services(case, ("0", "1", "2", "3"), "test"):
        command = service.command
        assert "--shutdown-timeout" not in command and "10" not in command
        assert "--compilation-config" not in command
        assert not any("cudagraph_mode" in token for token in command)
        assert "--enable-chunked-prefill" not in command
        assert "--enforce-eager" in command
        assert command[:2] == ("vllm", "serve")
        assert "--data-parallel-size" in command


def test_source_manifest_records_git_without_internal_source_hashes(make_case, tmp_path):
    internal = tmp_path / "afd_plugin/v1/worker/npu/ffn_model_runner.py"
    internal.parent.mkdir(parents=True)
    internal.write_text("not imported")
    manifest = runner.source_manifest(make_case("--repo", tmp_path))
    assert not any(path.startswith("afd-plugin/") for path in manifest["source_sha256"])
    assert manifest["afd_plugin"] is None


def test_metadata_does_not_inherit_a_parent_git_repository(make_case, monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *_a, **_k: pytest.fail("looked up parent Git")
    )
    manifest = runner.source_manifest(make_case("--repo", config.TOOL_ROOT / "tests"))
    assert manifest["afd_plugin"] is None


def test_native_baseline_can_configure_standard_dbo(make_case):
    case = make_case("--baseline", "--dbo", "on", "--mrv", "mrv2")
    services = build_services(case, ("0", "1", "2", "3"), "test")
    assert len(services) == 1
    assert "--enable-dbo" in services[0].command
    assert "--worker-extension-cls" not in services[0].command


def test_deployment_environment_can_override_version_presets(make_case, monkeypatch):
    monkeypatch.setenv("AFD_PLUGIN_EARLY_ENGINE_PATCH", "1")
    case = make_case(
        "--serve-env", "VLLM_PLUGINS=ascend,new_afd_entrypoint",
        "--serve-env", "AFD_NEW_OPTION=value=with=equals",
        "--serve-env", "VLLM_USE_V2_MODEL_RUNNER=1",
    )
    for service in build_services(case, ("0", "1", "2", "3"), "test"):
        env = runtime_environment(service)
        assert env["VLLM_PLUGINS"] == "ascend,new_afd_entrypoint"
        assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        assert env["AFD_NEW_OPTION"] == "value=with=equals"
        assert env["AFD_PLUGIN_EARLY_ENGINE_PATCH"] == "1"
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(service.devices)


@pytest.mark.parametrize("value", [
    "MISSING_VALUE", "0INVALID=yes", "ASCEND_RT_VISIBLE_DEVICES=8",
    "AFD_E2E_RUN_ID=other", "AFD_E2E_PROCESS_ROLE=other", "PYTHONPATH=/other",
])
def test_serve_environment_preserves_device_and_process_ownership(make_case, value):
    with pytest.raises(ValueError, match="serve environment"):
        make_case("--serve-env", value)
