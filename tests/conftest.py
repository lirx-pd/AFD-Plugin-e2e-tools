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
