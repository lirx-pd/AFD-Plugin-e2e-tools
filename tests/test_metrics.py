import io
import urllib.error
from http.client import IncompleteRead

import pytest

from afd_e2e import metrics


def config(*, baseline=False, topology="dp2,dp3"):
    return {
        "baseline": baseline,
        "topology": topology,
        "model": "v2-lite",
        "api_host": "0.0.0.0",
        "api_port": 8510,
    }


def test_capture_unique_http_instances(monkeypatch, tmp_path):
    calls = []
    payload = b'# HELP vllm:request_success Total requests\nvllm:request_success 7\n'

    class Opener:
        def open(self, url, *, timeout):
            calls.append((url, timeout))
            response = io.BytesIO(payload)
            response.status = 200
            return response

    def build_opener(handler):
        assert handler.proxies == {}
        return Opener()

    monkeypatch.setattr(metrics.urllib.request, "build_opener", build_opener)
    result = metrics.capture_metrics(config(baseline=True), tmp_path, "before")

    assert calls == [
        ("http://127.0.0.1:8510/metrics", 3),
        ("http://127.0.0.1:8511/metrics", 3),
    ]
    assert result == {
        "http://127.0.0.1:8510": {
            "status": "captured", "path": "metrics/before-0.prom",
        },
        "http://127.0.0.1:8511": {
            "status": "captured", "path": "metrics/before-1.prom",
        },
    }
    assert (tmp_path / "metrics/before-0.prom").read_bytes() == payload
    assert (tmp_path / "metrics/before-1.prom").read_bytes() == payload


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("http://localhost/metrics", 404, "Not Found", {}, None),
        urllib.error.URLError("endpoint unavailable"),
        TimeoutError("metrics timeout"),
    ],
    ids=["not_found", "unavailable", "timeout"],
)
def test_unavailable_metrics_are_nonfatal(monkeypatch, tmp_path, error):
    class Opener:
        def open(self, url, *, timeout):
            raise error

    monkeypatch.setattr(metrics.urllib.request, "build_opener", lambda *_: Opener())
    result = metrics.capture_metrics(config(), tmp_path, "after")
    assert result == {
        "http://127.0.0.1:8510": {"status": "unavailable", "error": str(error)},
    }
    assert not (tmp_path / "metrics").exists()


def test_non_200_endpoint_is_unavailable(monkeypatch, tmp_path):
    class Opener:
        def open(self, url, *, timeout):
            response = io.BytesIO(b"vllm:request_success 7\n")
            response.status = 503 if ":8510/" in url else 200
            return response

    monkeypatch.setattr(metrics.urllib.request, "build_opener", lambda *_: Opener())
    result = metrics.capture_metrics(config(baseline=True), tmp_path, "after")
    assert result["http://127.0.0.1:8510"] == {
        "status": "unavailable", "error": "HTTP 503",
    }
    assert result["http://127.0.0.1:8511"] == {
        "status": "captured", "path": "metrics/after-1.prom",
    }
    assert not (tmp_path / "metrics/after-0.prom").exists()
    assert (tmp_path / "metrics/after-1.prom").read_bytes() == b"vllm:request_success 7\n"


def test_truncated_http_body_is_nonfatal(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        status = 200

        def read(self):
            raise IncompleteRead(b"partial", 8)

    class Opener:
        def open(self, url, *, timeout):
            return Response()

    monkeypatch.setattr(metrics.urllib.request, "build_opener", lambda *_: Opener())
    result = metrics.capture_metrics(config(), tmp_path, "after")
    assert result["http://127.0.0.1:8510"]["status"] == "unavailable"
    assert "IncompleteRead" in result["http://127.0.0.1:8510"]["error"]
    assert not (tmp_path / "metrics").exists()


@pytest.mark.parametrize("operation", ["mkdir", "write"])
def test_filesystem_errors_propagate(monkeypatch, tmp_path, operation):
    class Opener:
        def open(self, url, *, timeout):
            response = io.BytesIO(b"vllm:request_success 7\n")
            response.status = 200
            return response

    monkeypatch.setattr(metrics.urllib.request, "build_opener", lambda *_: Opener())
    if operation == "mkdir":
        (tmp_path / "metrics").write_text("not a directory")
        expected_error = FileExistsError
    else:
        def fail_write(path, body):
            raise PermissionError("cannot write metrics")

        monkeypatch.setattr(metrics.Path, "write_bytes", fail_write)
        expected_error = PermissionError
    with pytest.raises(expected_error):
        metrics.capture_metrics(config(), tmp_path, "after")
