import json
import sys

import pytest

from afd_e2e.cli import parser
from afd_e2e.config import from_args


@pytest.fixture(autouse=True)
def clear_device_mask(monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)


@pytest.fixture
def make_case():
    def make(*arguments):
        return from_args(parser().parse_args(["run", *map(str, arguments)]))

    return make


@pytest.fixture
def manager(tmp_path):
    script = tmp_path / "device manager.py"
    record = tmp_path / "released.json"
    script.write_text("""import json, pathlib, sys
payload = json.load(sys.stdin)
if sys.argv[1] == "allocate":
    assert int(sys.argv[3]) == payload["count"]
    response = {"lease_id": "lease-123", "devices": list(range(payload["count"]))}
    if sys.argv[2] == "duplicate":
        response["devices"] = [0] * payload["count"]
    if sys.argv[2] == "short":
        response["devices"].pop()
    if sys.argv[2] == "wrong_type":
        response["devices"] = None
    print(json.dumps(response))
else:
    assert sys.argv[3] == payload["lease_id"]
    pathlib.Path(sys.argv[2]).write_text(json.dumps(payload))
""")

    def platform(mode="valid"):
        path = tmp_path / "platform.json"
        path.write_text(
            json.dumps(
                {
                    "allocator": {
                        "allocate": [
                            sys.executable,
                            str(script),
                            "allocate",
                            mode,
                            "{count}",
                        ],
                        "release": [
                            sys.executable,
                            str(script),
                            "release",
                            str(record),
                            "{lease_id}",
                        ],
                        "timeout": 5,
                    }
                }
            )
        )
        return path, record

    return platform
