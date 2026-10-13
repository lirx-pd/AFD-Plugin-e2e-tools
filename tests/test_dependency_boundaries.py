import json
import subprocess
import sys
from pathlib import Path

import pytest


ISOLATED_COMMAND = """
import importlib.abc
import runpy
import sys

blocked = (
    "torch", "torch_npu", "vllm", "vllm_ascend", "afd_plugin",
    "httpx", "lm_eval", "datasets", "pyarrow", "yaml",
    "afd_e2e.observer", "afd_e2e.coverage",
)
if sys.argv[2] != "import-only":
    blocked += ("afd_e2e.runner", "afd_e2e.probe", "afd_e2e.evaluation")

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ImportError("Unexpected runtime dependency: " + fullname)

sys.meta_path.insert(0, BlockRuntime())
sys.path.insert(0, sys.argv[1])
module = sys.argv[2]
if module == "import-only":
    import afd_e2e.runner
    import afd_e2e.probe
    import afd_e2e.metrics
    import afd_e2e.performance
    import afd_e2e.workloads
else:
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name="__main__")
"""


@pytest.mark.parametrize(
    "module,arguments",
    [
        ("afd_e2e.cli", ["--help"]),
        ("afd_e2e.cli", ["run", "--help"]),
        ("afd_e2e.cli", ["plan", "--output", "{plan}"]),
        ("afd_e2e.cli", ["plan", "--baseline", "--topology", "dp1,dp2"]),
        ("afd_e2e.cli", ["status", "{run}", "--json"]),
        ("afd_e2e.cli", ["pack", "{run}", "--compact"]),
        ("afd_e2e.performance", ["--help"]),
        ("afd_e2e.workloads", ["--help"]),
    ],
)
def test_offline_and_http_commands_do_not_import_runtime(tmp_path, module, arguments):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "status": "passed", "finished_at": "2026-10-12", "case_id": "offline",
    }))
    replacements = {"run": str(run), "plan": str(tmp_path / "plan")}
    result = subprocess.run(
        [sys.executable, "-I", "-c", ISOLATED_COMMAND,
         str(Path(__file__).resolve().parents[1]), module,
         *(arg.format(**replacements) for arg in arguments)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    if "{plan}" in arguments:
        assert (tmp_path / "plan/plan.json").is_file()
    if "pack" in arguments:
        assert (run / "reports.tar.gz").is_file()


def test_run_loads_runtime_orchestration():
    result = subprocess.run(
        [sys.executable, "-I", "-c", ISOLATED_COMMAND,
         str(Path(__file__).resolve().parents[1]), "afd_e2e.cli",
         "run", "--baseline", "--device", "0,1,2,3"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "Unexpected runtime dependency: afd_e2e.runner" in result.stderr


def test_orchestration_imports_no_upstream_or_private_observers():
    result = subprocess.run(
        [sys.executable, "-I", "-c", ISOLATED_COMMAND,
         str(Path(__file__).resolve().parents[1]), "import-only"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
