import errno
import socket
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_busy_port_fails_before_loading_npu_environment():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        args = SimpleNamespace(api_host="127.0.0.1", api_port=port, afd_port=port, ports=None)
        with pytest.raises(OSError) as error:
            probe(args)
        assert error.value.errno == errno.EADDRINUSE


@pytest.mark.parametrize("async_cam", [False, True])
def test_missing_cam_ops_fail_preflight(monkeypatch, tmp_path, async_cam):
    import afd_e2e.probe as module

    monkeypatch.setattr(module, "check_ports", lambda *_: None)
    monkeypatch.setattr(
        module.importlib.metadata,
        "entry_points",
        lambda **_: [SimpleNamespace(name="afd")],
    )
    torch = MagicMock()
    torch.npu.device_count.return_value = 1
    torch.testing.assert_close = MagicMock()
    plugin = SimpleNamespace(__file__=str(tmp_path / "afd_plugin" / "__init__.py"))

    ops = SimpleNamespace(
        ensure_cam_p2p_ops_available=MagicMock(
            side_effect=RuntimeError("sync CAM custom ops not available")
        ),
        ensure_cam_async_ops_available=MagicMock(
            side_effect=RuntimeError("async CAM custom ops not available")
        ),
    )
    for name, value in {
        "torch": torch,
        "torch_npu": SimpleNamespace(),
        "vllm": SimpleNamespace(),
        "vllm_ascend": SimpleNamespace(),
        "afd_plugin": plugin,
        "afd_plugin.compat.npu.ops": ops,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    model = tmp_path / "weights"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"deepseek_v2"}')
    (model / "tokenizer_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture")
    args = SimpleNamespace(
        api_host="127.0.0.1",
        api_port=12345,
        afd_port=6455,
        ports=None,
        nic=None,
        repo=str(tmp_path),
        devices="0",
        model=str(model),
        family="v2-lite",
        baseline=False,
        async_cam=async_cam,
        layered=False,
    )
    mode = "async" if async_cam else "sync"
    with pytest.raises(RuntimeError, match=f"^{mode} CAM custom ops not available"):
        probe(args)


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
