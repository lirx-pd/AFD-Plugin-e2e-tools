# SPDX-License-Identifier: Apache-2.0
"""Test-only worker observations. No model inputs, outputs or plans are changed."""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from functools import wraps

EVIDENCE_PREFIX = "AFD_E2E_EVENT "


class Worker:
    """vLLM import-only extension, loaded only by this harness's workers."""


def install() -> None:
    from afd_plugin.v1.worker.attention_metadata import AFDMetadataProviderMixin
    from afd_plugin.v1.worker import dbo
    from afd_plugin.connectors.npu.camp2p import CAMP2pAFDConnector
    from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
    from afd_plugin.v1.worker.npu.attention_model_runner import (
        AFDNPUAttentionModelRunner,
    )
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )
    from afd_plugin.v1.worker.npu import ffn_model_runner as ffn_module
    from afd_plugin.model_executor.npu.async_cam_w4a8 import AsyncCAMW4A8Executor
    from vllm.distributed import get_dp_group, get_tp_group
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker import ubatching

    def emit(runner, kind, **values):
        event = {
            "time": time.time(),
            "run_id": os.environ["AFD_E2E_RUN_ID"],
            "role": os.environ["AFD_E2E_PROCESS_ROLE"],
            "kind": kind,
            "dp_rank": get_dp_group().rank_in_group,
            "tp_rank": get_tp_group().rank_in_group,
            "class": type(runner).__module__ + "." + type(runner).__name__,
            **values,
        }
        sys.stdout.write(
            EVIDENCE_PREFIX + json.dumps(event, separators=(",", ":")) + "\n"
        )
        sys.stdout.flush()

    def observe_metadata(runner):
        state = runner._afd_e2e_state
        if state is None:
            return
        metadata = runner._afd_pending_metadata
        if metadata is not None:
            state["stages"] = metadata.num_stages
        else:
            slices = getattr(runner, "ubatch_slices", None)
            state["stages"] = len(slices) if slices else 1
        if runner._afd_is_graph_replaying:
            state["mode"] = "FULL"

    def wrap_attention(cls):
        native_init = cls.__init__
        native_execute = cls.execute_model

        @wraps(native_init)
        def initialize(self, *args, **kwargs):
            native_init(self, *args, **kwargs)
            self._afd_e2e_state = None
            self._afd_e2e_count = 0
            self._afd_e2e_last = None
            self._afd_e2e_native_shape = None
            emit(self, "runner", mrv_flag=int(self.vllm_config.use_v2_model_runner))

        @wraps(native_execute)
        def execute(self, scheduler_output, *args, **kwargs):
            live = (
                scheduler_output.total_num_scheduled_tokens > 0
                and not kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
                and not kwargs.get("is_profile", args[3] if len(args) > 3 else False)
            )
            previous = self._afd_e2e_state
            self._afd_e2e_state = {"stages": 0, "mode": "eager"} if live else None
            try:
                result = native_execute(self, scheduler_output, *args, **kwargs)
                if live:
                    self._afd_e2e_count += 1
                    feature_state = (
                        self._afd_e2e_state["stages"],
                        self._afd_e2e_state["mode"],
                    )
                    # Every feature-state change is reported; steady execution is
                    # sampled to keep multi-million-token runs small.
                    if (
                        self._afd_e2e_count % 128 in (1, 2)
                        or feature_state != self._afd_e2e_last
                    ):
                        emit(
                            self,
                            "execution",
                            phase="live",
                            tokens=int(scheduler_output.total_num_scheduled_tokens),
                            count=self._afd_e2e_count,
                            **self._afd_e2e_state,
                        )
                    self._afd_e2e_last = feature_state
                return result
            finally:
                self._afd_e2e_state = previous

        cls.__init__ = initialize
        cls.execute_model = execute

    def wrap_metadata(cls, name):
        native = getattr(cls, name)

        @wraps(native)
        def observe(self, *args, **kwargs):
            result = native(self, *args, **kwargs)
            observe_metadata(self)
            return result

        setattr(cls, name, observe)

    wrap_attention(AFDNPUAttentionModelRunner)
    wrap_attention(AFDNPUAttentionModelRunnerV2)
    wrap_metadata(AFDNPUAttentionModelRunner, "_model_forward")
    wrap_metadata(AFDNPUAttentionModelRunner, "_send_dp_metadata")
    wrap_metadata(AFDMetadataProviderMixin, "install_afd_metadata_on_forward_context")

    native_install_stages = AFDMetadataProviderMixin.install_mrv2_ubatch_metadata

    @wraps(native_install_stages)
    def install_stages(self, slices, contexts, real_tokens):
        native_install_stages(self, slices, contexts, real_tokens)
        observe_metadata(self)
        if self._afd_e2e_state is None:
            return
        capacity = [stage.num_tokens for stage in slices]
        real = [
            context.additional_kwargs["afd_metadata"].tokens_unpadded_lens[0]
            for context in contexts
        ]
        for context in contexts:
            context.additional_kwargs["_afd_e2e_trace"] = self._afd_e2e_count in (0, 1)
        # Preserve rare padding-only and odd-split evidence between samples.
        shape = (0 in real, sum(capacity) % 2, capacity != real)
        if self._afd_e2e_count % 128 in (0, 1) or shape != self._afd_e2e_native_shape:
            emit(
                self,
                "native_ubatches",
                phase="live",
                executor=type(self.ubatch_runner).__name__,
                capacity=capacity,
                real=real,
            )
        self._afd_e2e_native_shape = shape

    AFDMetadataProviderMixin.install_mrv2_ubatch_metadata = install_stages

    def handoff(action, context):
        if context.additional_kwargs.get("_afd_e2e_trace", False):
            metadata = context.additional_kwargs["afd_metadata"]
            emit(
                metadata.connector,
                "handoff",
                phase="live",
                action=action,
                stage=metadata.stage_idx,
                layer=context.additional_kwargs["_afd_e2e_layer"],
            )

    native_send = CAMP2pAFDConnector.send_attn_output
    native_recv = CAMP2pAFDConnector.recv_ffn_output
    native_yield = dbo._yield_if_dbo_enabled

    @wraps(native_send)
    def send(self, hidden_states, context, **kwargs):
        result = native_send(self, hidden_states, context, **kwargs)
        if ubatching.dbo_enabled():
            forward = get_forward_context()
            forward.additional_kwargs["_afd_e2e_layer"] = context.metadata.layer_idx
            handoff("send_return", forward)
        return result

    @wraps(native_recv)
    def receive(self, *args, **kwargs):
        result = native_recv(self, *args, **kwargs)
        if ubatching.dbo_enabled():
            handoff("recv_return", get_forward_context())
        return result

    @wraps(native_yield)
    def yield_to_peer():
        if not ubatching.dbo_enabled():
            return native_yield()
        owner = get_forward_context()
        handoff("yield", owner)
        result = native_yield()
        handoff("resume", owner)
        return result

    CAMP2pAFDConnector.send_attn_output = send
    CAMP2pAFDConnector.recv_ffn_output = receive
    dbo._yield_if_dbo_enabled = yield_to_peer

    native_cam_send = CAMAsyncAFDConnector.send_attn_output
    native_cam_recv = CAMAsyncAFDConnector.recv_ffn_output

    @wraps(native_cam_send)
    def cam_send(self, hidden_states, context, **kwargs):
        result = native_cam_send(self, hidden_states, context, **kwargs)
        if ubatching.dbo_enabled():
            forward = get_forward_context()
            forward.additional_kwargs["_afd_e2e_layer"] = context.metadata.layer_idx
            handoff("send_return", forward)
        return result

    @wraps(native_cam_recv)
    def cam_receive(self, *args, **kwargs):
        result = native_cam_recv(self, *args, **kwargs)
        transfer = kwargs.get("context")
        if (
            transfer is not None
            and ubatching.dbo_enabled()
            and get_forward_context().additional_kwargs.get("_afd_e2e_trace", False)
        ):
            emit(
                self,
                "cam_raw_receive",
                phase="live",
                owner_stage=transfer.metadata.stage_idx,
                actor_stage=ubatching._THREAD_ID_TO_CONTEXT.get(threading.get_ident()),
                layer=transfer.metadata.layer_idx,
            )
        return result

    CAMAsyncAFDConnector.send_attn_output = cam_send
    CAMAsyncAFDConnector.recv_ffn_output = cam_receive

    cls = ffn_module.AFDNPUFFNModelRunner
    native_initialize = cls._initialize_layered_executor
    native_execute = cls.execute_model
    native_async_forward = cls._ffn_forward_connector_driven
    native_mode = ffn_module.graph_run_mode
    native_layered_call = AsyncCAMW4A8Executor.__call__
    active = None
    layered_owner = None

    @wraps(native_initialize)
    def initialize_ffn(self):
        nonlocal layered_owner
        result = native_initialize(self)
        self._afd_e2e_count = 0
        self._afd_e2e_mode = "eager"
        self._afd_e2e_last = None
        self._afd_e2e_layered_calls = 0
        actual = self._layered_executor is not None
        expected = os.environ["AFD_E2E_LAYERED_EXPECTED"] == "1"
        if actual != expected:
            raise RuntimeError(f"Requested layered={expected}, actual layered={actual}")
        layered_owner = self
        emit(
            self,
            "runner",
            mrv_flag=int(self.vllm_config.use_v2_model_runner),
            layered=actual,
        )
        return result

    @wraps(native_mode)
    def observe_mode(**kwargs):
        result = native_mode(**kwargs)
        if active is not None:
            active._afd_e2e_mode = result.value
        return result

    @wraps(native_execute)
    def execute_ffn(self, *args, **kwargs):
        nonlocal active
        previous = active
        active = self
        try:
            result = native_execute(self, *args, **kwargs)
            if not any(
                kwargs.get(name, False)
                for name in ("is_graph_capturing", "is_warmup", "is_profile")
            ):
                self._afd_e2e_count += 1
                feature_state = (len(kwargs["dp_metadata_list"]), self._afd_e2e_mode)
                if (
                    self._afd_e2e_count % 128 in (1, 2)
                    or feature_state != self._afd_e2e_last
                ):
                    emit(
                        self,
                        "execution",
                        phase="live",
                        mode=self._afd_e2e_mode,
                        stages=feature_state[0],
                        count=self._afd_e2e_count,
                    )
                self._afd_e2e_last = feature_state
            return result
        finally:
            active = previous

    @wraps(native_async_forward)
    def async_forward(self, *args, **kwargs):
        result = native_async_forward(self, *args, **kwargs)
        self._afd_e2e_count += 1
        if self._afd_e2e_count % 128 in (1, 2):
            emit(
                self,
                "execution",
                phase="live",
                mode="eager",
                stages=None,
                count=self._afd_e2e_count,
            )
        return result

    @wraps(native_layered_call)
    def layered_call(self, *args, **kwargs):
        result = native_layered_call(self, *args, **kwargs)
        if layered_owner is not None:
            layered_owner._afd_e2e_layered_calls += 1
            if layered_owner._afd_e2e_layered_calls % 128 in (1, 2):
                emit(layered_owner, "layered", phase="live")
        return result

    cls._initialize_layered_executor = initialize_ffn
    cls.execute_model = execute_ffn
    cls._ffn_forward_connector_driven = async_forward
    ffn_module.graph_run_mode = observe_mode
    AsyncCAMW4A8Executor.__call__ = layered_call


install()
