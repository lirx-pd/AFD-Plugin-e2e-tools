import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def observed_runners(monkeypatch):
    def module(name, **attributes):
        if name in sys.modules:
            result = sys.modules[name]
        else:
            result = ModuleType(name)
            result.__path__ = []
            monkeypatch.setitem(sys.modules, name, result)
            if "." in name:
                parent, child = name.rsplit(".", 1)
                monkeypatch.setattr(module(parent), child, result, raising=False)
        result.__dict__.update(attributes)
        return result

    sentinel = object()

    class MetadataProvider:
        def install_afd_metadata_on_forward_context(self):
            return sentinel

        def install_mrv2_ubatch_metadata(self, slices, contexts, real_tokens):
            self.native_stage_args = (slices, contexts, real_tokens)

    def initialize(self, mrv):
        self.vllm_config = SimpleNamespace(use_v2_model_runner=mrv)
        self._afd_pending_metadata = SimpleNamespace(num_stages=2)
        self._afd_is_graph_replaying = False

    def metadata(self):
        self._afd_is_graph_replaying = True
        return self.install_afd_metadata_on_forward_context()

    def execute(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
        **kwargs,
    ):
        self._model_forward()
        if kwargs.get("fail"):
            raise RuntimeError("native failure")
        return sentinel

    def attention(name):
        return type(
            name,
            (MetadataProvider,),
            {
                "__init__": initialize,
                "execute_model": execute,
                "_model_forward": metadata,
                "_send_dp_metadata": metadata,
            },
        )

    v1 = attention("AFDNPUAttentionModelRunner")
    v2 = attention("AFDNPUAttentionModelRunnerV2")
    module(
        "afd_plugin.v1.worker.attention_metadata",
        AFDMetadataProviderMixin=MetadataProvider,
    )
    module(
        "afd_plugin.v1.worker.npu.attention_model_runner", AFDNPUAttentionModelRunner=v1
    )
    module(
        "afd_plugin.v1.worker.npu.attention_model_runner_v2",
        AFDNPUAttentionModelRunnerV2=v2,
    )

    class Layered:
        def __call__(self):
            return sentinel

    class FFN:
        def __init__(self, mrv, layered):
            self.vllm_config = SimpleNamespace(use_v2_model_runner=mrv)
            self._layered_executor = Layered() if layered else None

        def _initialize_layered_executor(self):
            return sentinel

        def execute_model(self, **kwargs):
            ffn_module.graph_run_mode(**kwargs)
            return sentinel

        def _ffn_forward_connector_driven(self):
            if self._layered_executor:
                self._layered_executor()
            return sentinel

    ffn_module = module(
        "afd_plugin.v1.worker.npu.ffn_model_runner",
        AFDNPUFFNModelRunner=FFN,
        graph_run_mode=lambda **kw: SimpleNamespace(
            value="replay" if kw.get("is_graph_replaying") else "eager"
        ),
    )
    module("afd_plugin.model_executor.npu.async_cam_w4a8", AsyncCAMW4A8Executor=Layered)
    module(
        "vllm.distributed",
        get_dp_group=lambda: SimpleNamespace(rank_in_group=0),
        get_tp_group=lambda: SimpleNamespace(rank_in_group=0),
    )
    module("vllm.v1.worker.ubatching", dbo_enabled=lambda: False)
    forward_context = module("vllm.forward_context", current_context=None)
    forward_context.get_forward_context = lambda: forward_context.current_context
    module("afd_plugin.v1.worker.dbo", _yield_if_dbo_enabled=lambda: sentinel)
    module(
        "afd_plugin.connectors.npu.camp2p",
        CAMP2pAFDConnector=type(
            "CAMP2pAFDConnector",
            (),
            {
                "send_attn_output": lambda *args, **kw: sentinel,
                "recv_ffn_output": lambda *args, **kw: sentinel,
            },
        ),
    )
    module(
        "afd_plugin.connectors.npu.async_cam",
        CAMAsyncAFDConnector=type(
            "CAMAsyncAFDConnector",
            (),
            {
                "send_attn_output": lambda *a, **k: sentinel,
                "recv_ffn_output": lambda *a, **k: sentinel,
            },
        ),
    )
    monkeypatch.setenv("AFD_E2E_RUN_ID", "observer-test")
    path = Path(__file__).parents[1] / "afd_e2e/observer.py"
    spec = importlib.util.spec_from_file_location("observer_under_test", path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    return v1, v2, FFN, sentinel


def events(capsys):
    return [
        json.loads(line.split("AFD_E2E_EVENT ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
    ]


@pytest.mark.parametrize("mrv", [0, 1])
def test_attention_observer_preserves_outputs_and_excludes_dummy(
    observed_runners, monkeypatch, capsys, mrv
):
    v1, v2, _, sentinel = observed_runners
    monkeypatch.setenv("AFD_E2E_PROCESS_ROLE", "attention")
    runner = (v2 if mrv else v1)(mrv)
    scheduled = SimpleNamespace(total_num_scheduled_tokens=10)
    assert events(capsys)[0]["mrv_flag"] == mrv
    assert runner.execute_model(scheduled, None, True) is sentinel
    assert runner.execute_model(scheduled, is_profile=True) is sentinel
    assert events(capsys) == []
    assert runner.execute_model(scheduled) is sentinel
    evidence = events(capsys)[0]
    assert (evidence["stages"], evidence["mode"], evidence["phase"]) == (
        2,
        "FULL",
        "live",
    )
    assert runner._afd_e2e_state is None
    with pytest.raises(RuntimeError, match="native failure"):
        runner.execute_model(scheduled, fail=True)
    assert runner._afd_e2e_state is None
    assert events(capsys) == []


@pytest.mark.parametrize("mrv", [0, 1])
def test_ffn_layered_observed_for_both_mrv_flags(
    observed_runners, monkeypatch, capsys, mrv
):
    _, _, FFN, sentinel = observed_runners
    monkeypatch.setenv("AFD_E2E_PROCESS_ROLE", "ffn")
    monkeypatch.setenv("AFD_E2E_LAYERED_EXPECTED", "1")
    runner = FFN(mrv, layered=True)
    assert runner._initialize_layered_executor() is sentinel
    assert events(capsys)[0]["mrv_flag"] == mrv
    assert runner._ffn_forward_connector_driven() is sentinel
    assert {event["kind"] for event in events(capsys)} == {"layered", "execution"}


def test_ffn_graph_observer_excludes_warmup(observed_runners, monkeypatch, capsys):
    _, _, FFN, sentinel = observed_runners
    monkeypatch.setenv("AFD_E2E_PROCESS_ROLE", "ffn")
    monkeypatch.setenv("AFD_E2E_LAYERED_EXPECTED", "0")
    runner = FFN(0, layered=False)
    runner._initialize_layered_executor()
    events(capsys)
    assert (
        runner.execute_model(dp_metadata_list={0: None, 1: None}, is_warmup=True)
        is sentinel
    )
    assert events(capsys) == []
    assert (
        runner.execute_model(
            dp_metadata_list={0: None, 1: None}, is_graph_replaying=True
        )
        is sentinel
    )
    evidence = events(capsys)[0]
    assert (evidence["mode"], evidence["stages"]) == ("replay", 2)


def test_native_stage_trace_distinguishes_capacity_and_real_tokens(
    observed_runners, monkeypatch, capsys
):
    _, v2, _, _ = observed_runners
    monkeypatch.setenv("AFD_E2E_PROCESS_ROLE", "attention")
    runner = v2(1)
    runner.ubatch_runner = SimpleNamespace()
    stages = [SimpleNamespace(num_tokens=4) for _ in range(2)]
    contexts = [
        SimpleNamespace(
            additional_kwargs={
                "afd_metadata": SimpleNamespace(tokens_unpadded_lens=[n])
            }
        )
        for n in (4, 1)
    ]
    events(capsys)
    runner.install_mrv2_ubatch_metadata(stages, contexts, 5)
    assert events(capsys) == []
    runner._afd_e2e_state = {"stages": 0, "mode": "eager"}
    runner.install_mrv2_ubatch_metadata(stages, contexts, 5)
    assert runner.native_stage_args == (stages, contexts, 5)
    evidence = events(capsys)[0]
    assert evidence["kind"] == "native_ubatches"
    assert evidence["capacity"] == [4, 4]
    assert evidence["real"] == [4, 1]
    runner._afd_e2e_count = 33
    runner.install_mrv2_ubatch_metadata(stages, contexts, 5)
    assert events(capsys) == []
    contexts[1].additional_kwargs["afd_metadata"].tokens_unpadded_lens = [0]
    runner.install_mrv2_ubatch_metadata(stages, contexts, 4)
    assert events(capsys)[0]["real"] == [4, 0]


def test_native_handoff_trace_preserves_owner_and_outputs(
    observed_runners, monkeypatch, capsys
):
    _, _, _, sentinel = observed_runners
    from afd_plugin.connectors.npu.camp2p import CAMP2pAFDConnector
    from afd_plugin.v1.worker import dbo
    from vllm.v1.worker import ubatching
    from vllm import forward_context

    monkeypatch.setenv("AFD_E2E_PROCESS_ROLE", "attention")
    connector = CAMP2pAFDConnector()
    owner = SimpleNamespace(
        additional_kwargs={
            "_afd_e2e_trace": True,
            "afd_metadata": SimpleNamespace(stage_idx=0, connector=connector),
        }
    )
    monkeypatch.setattr(ubatching, "dbo_enabled", lambda: True)
    monkeypatch.setattr(forward_context, "current_context", owner)
    transfer = SimpleNamespace(metadata=SimpleNamespace(layer_idx=42))
    assert connector.send_attn_output(None, transfer) is sentinel
    assert dbo._yield_if_dbo_enabled() is sentinel
    assert connector.recv_ffn_output(None, ubatch_idx=0) is sentinel
    trace = events(capsys)
    assert [x["action"] for x in trace] == [
        "send_return",
        "yield",
        "resume",
        "recv_return",
    ]
    assert all(x["stage"] == 0 and x["layer"] == 42 for x in trace)
