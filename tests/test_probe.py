import builtins
import errno
import hashlib
import json
import socket
import sys
from types import SimpleNamespace

import pytest

from afd_e2e.probe import check_ports, probe


def test_port_check_rejects_existing_listener():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        with pytest.raises(OSError) as error:
            check_ports("127.0.0.1", (server.getsockname()[1],))
        assert error.value.errno == errno.EADDRINUSE


def test_busy_port_fails_before_reading_package_metadata(monkeypatch):
    import afd_e2e.probe as module

    def fail_if_called(_):
        raise AssertionError("Package discovery must follow port checks")

    monkeypatch.setattr(module.importlib.metadata, "version", fail_if_called)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        args = SimpleNamespace(api_host="127.0.0.1", ports=[port])
        with pytest.raises(OSError) as error:
            probe(args)
        assert error.value.errno == errno.EADDRINUSE


@pytest.fixture
def probe_environment(monkeypatch):
    import afd_e2e.probe as module

    monkeypatch.setattr(module, "check_ports", lambda *_: None)
    monkeypatch.setattr(
        module.importlib.metadata, "version", lambda name: "test-version"
    )
    return SimpleNamespace(
        api_host="127.0.0.1",
        ports=[8500, 8501, 6455],
        nic=None,
        devices="0,1",
        model="organization/remote-model",
    )


def local_model(tmp_path):
    model = tmp_path / "weights"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"deepseek_v2","num_hidden_layers":27}'
    )
    (model / "tokenizer_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture")
    return model


@pytest.mark.parametrize("local", [False, True])
def test_probe_never_imports_upstream_packages(
    monkeypatch, tmp_path, probe_environment, local
):
    import afd_e2e.probe as module

    original_import = builtins.__import__
    blocked = {
        "torch", "torch_npu", "vllm", "vllm_ascend", "afd_plugin", "transformers"
    }

    def reject_upstream(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise AssertionError(f"Upstream import attempted: {name}")
        return original_import(name, *args, **kwargs)

    def reject_entry_points(**_):
        raise AssertionError("Plugin entry points must be handled by vllm serve")

    if local:
        probe_environment.model = str(local_model(tmp_path))
    monkeypatch.setattr(builtins, "__import__", reject_upstream)
    monkeypatch.setattr(module.importlib.metadata, "entry_points", reject_entry_points)
    result = probe(probe_environment)
    assert result["versions"]["vllm-afd-plugin"] == "test-version"
    assert result["requested_devices"] == ["0", "1"]
    assert "tested_devices" not in result
    assert "afd_source" not in result
    assert "vllm_source" not in result
    if local:
        assert result["model_type"] == "deepseek_v2"
        assert result["layers"] == 27
        assert result["weight_shards"] == 1
        assert result["model_hashes"]["tokenizer_config.json"] == hashlib.sha256(
            b"{}"
        ).hexdigest()
    else:
        assert result["model_type"] is None
        assert result["layers"] is None
        assert result["weight_shards"] == 0
        assert result["model_hashes"] == {}


def test_missing_distribution_is_recorded_without_rejecting_run(
    monkeypatch, probe_environment
):
    import afd_e2e.probe as module

    def version(name):
        if name == "vllm-afd-plugin":
            raise module.importlib.metadata.PackageNotFoundError(name)
        return "recorded-version"

    monkeypatch.setattr(module.importlib.metadata, "version", version)
    result = probe(probe_environment)
    assert result["versions"]["vllm-afd-plugin"] is None
    assert result["versions"]["vllm"] == "recorded-version"


@pytest.mark.parametrize("problem", [
    "missing-config", "invalid-json", "no-weights", "empty-shard",
    "missing-shard", "outside-shard",
])
def test_local_model_integrity_failures(tmp_path, probe_environment, problem):
    model = local_model(tmp_path)
    probe_environment.model = str(model)
    expected_error = RuntimeError
    if problem == "missing-config":
        (model / "config.json").unlink()
        expected_error = FileNotFoundError
    elif problem == "invalid-json":
        (model / "config.json").write_text("{")
        expected_error = json.JSONDecodeError
    elif problem == "no-weights":
        (model / "model.safetensors").unlink()
    elif problem == "empty-shard":
        (model / "model.safetensors").write_bytes(b"")
    else:
        name = "missing.safetensors"
        if problem == "outside-shard":
            name = "../outside.safetensors"
            (tmp_path / "outside.safetensors").write_bytes(b"fixture")
        else:
            expected_error = FileNotFoundError
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": name}})
        )
    with pytest.raises(expected_error):
        probe(probe_environment)


def test_local_tokenizer_metadata_is_optional(tmp_path, probe_environment):
    model = local_model(tmp_path)
    (model / "tokenizer_config.json").unlink()
    probe_environment.model = str(model)
    assert probe(probe_environment)["weight_shards"] == 1


def test_unknown_network_interface_fails_preflight(monkeypatch, probe_environment):
    probe_environment.nic = "missing-interface"

    def unknown_interface(name):
        assert name == "missing-interface"
        raise OSError("No interface with this name")

    monkeypatch.setattr(socket, "if_nametoindex", unknown_interface)
    with pytest.raises(OSError, match="No interface"):
        probe(probe_environment)


def test_probe_cli_accepts_only_external_preflight_parameters(
    monkeypatch, probe_environment, capsys
):
    import afd_e2e.probe as module

    checked = []
    monkeypatch.setattr(
        module, "check_ports", lambda host, ports: checked.append((host, ports))
    )
    monkeypatch.setattr(sys, "argv", [
        "probe", "--model", probe_environment.model, "--devices", "2,3",
        "--api-host", "127.0.0.1", "--ports", "8500,8501,6455",
    ])
    module.main()
    result = json.loads(capsys.readouterr().out.removeprefix("AFD_E2E_PROBE "))
    assert result["requested_devices"] == ["2", "3"]
    assert checked == [("127.0.0.1", (8500, 8501, 6455))]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux server TIME_WAIT")
def test_port_check_allows_previous_run_time_wait():
    with socket.socket() as server, socket.socket() as client:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        host, port = server.getsockname()
        server.listen()
        client.connect((host, port))
        connection, _ = server.accept()
        with connection:
            connection.sendall(b"x")
            assert client.recv(1) == b"x"
            connection.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
    with socket.socket() as endpoint, pytest.raises(OSError) as error:
        endpoint.bind((host, port))
    assert error.value.errno == errno.EADDRINUSE
    check_ports(host, (port,))
