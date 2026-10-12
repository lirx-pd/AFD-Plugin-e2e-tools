import asyncio
import io
import json
import statistics
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest

from afd_e2e import performance
from afd_e2e.planner import describe


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.closed = False
        self.exhausted = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk
        self.exhausted = True

    async def aclose(self):
        self.closed = True


def sse(values):
    return [
        b"data: [DONE]\n\n"
        if v == "[DONE]"
        else ("data: " + json.dumps(v, ensure_ascii=False) + "\n\n").encode()
        for v in values
    ]


def stream_values(input_tokens=8, output_tokens=4):
    return [
        {"choices": [{"text": "", "token_ids": [], "finish_reason": None}]},
        {
            "choices": [
                {"text": "你\u2028好", "token_ids": [1, 2], "finish_reason": None}
            ]
        },
        {"choices": [{"text": "end", "token_ids": [3, 4], "finish_reason": None}]},
        {"choices": [{"text": "", "token_ids": [], "finish_reason": "length"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
        },
        "[DONE]",
    ]


def test_plan_covers_lengths_concurrency_rates_and_mixed(make_case):
    case = make_case(
        "--test",
        "perf",
        "--perf-input-lens",
        "128,512",
        "--perf-output-lens",
        "1,64",
        "--perf-concurrencies",
        "1,8",
        "--rps",
        "2,inf",
        "--perf-weights",
        "8,1",
    )
    planned = describe(case)
    pts = planned["performance_points"]
    assert len(pts) == 20
    assert len({p["id"] for p in pts}) == 20
    assert sum(p["mode"] == "matrix" for p in pts) == 16
    assert all(p["weights"] == (8, 1) for p in pts if p["mode"] == "mixed")
    assert {p["rps"] for p in pts} == {2, None}
    json.dumps(planned, allow_nan=False)


@pytest.mark.parametrize(
    "flags",
    [
        ["--perf-input-lens", "0"],
        ["--perf-input-lens", "128,128"],
        ["--perf-output-lens", "-1"],
        ["--perf-concurrencies", "0"],
        ["--rps", "nan"],
        ["--rps", "0"],
        ["--rps=-inf"],
        ["--rps", "1,1"],
        ["--perf-weights", "1"],
        ["--perf-weights", "1,nan,1"],
        ["--perf-weights", "1,-1,1"],
        ["--perf-requests", "0"],
        ["--perf-warmup", "-1"],
        ["--perf-input-lens", "4096"],
        ["--perf-input-lens", ""],
    ],
)
def test_bad_performance_config_fails_before_allocation(make_case, flags):
    with pytest.raises(ValueError):
        make_case("--test", "perf", *flags)


def test_ordinary_runs_do_not_validate_unused_performance_length(make_case):
    assert make_case("--max-model-len", "128").max_model_len == 128


def test_arrival_rates_and_bursts_are_reproducible():
    assert performance.arrivals(5, 2, 3, "constant") == [0, 0.5, 1, 1.5, 2]
    assert performance.arrivals(5, 2, 3, "burst") == [0, 0, 0, 1.5, 1.5]
    assert performance.arrivals(5, None, 3, "burst") == [0] * 5
    offsets = performance.arrivals(10000, 2, 3, "poisson")
    assert offsets == performance.arrivals(10000, 2, 3, "poisson")
    assert statistics.mean(
        b - a for a, b in zip(offsets, offsets[1:])
    ) == pytest.approx(0.5, abs=0.03)


def test_percentiles_empty_single_and_multiple_samples():
    result = performance.distribution([4, 1, 3, 2])
    assert result["mean"] == result["p50"] == 2.5
    assert result["p95"] == pytest.approx(3.85)
    assert result["p99"] == pytest.approx(3.97)
    assert performance.distribution([])["mean"] is None
    assert performance.distribution([7])["p99"] == 7


def test_stream_uses_token_count_not_chunk_count_and_ignores_control_events(
    monkeypatch,
):
    chunks = sse(stream_values())
    payload = b": ping\n\n" + b"".join(chunks)
    pieces = [payload[i : i + 3] for i in range(0, len(payload), 3)]
    clock = iter([10.1, 10.3])
    stream = Chunks(pieces)
    monkeypatch.setattr(
        performance, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        ) as client:
            record = {"sent_at": 10}
            await performance.stream_request(
                client,
                "http://test/v1/completions",
                {"prompt": [1] * 8, "max_tokens": 4},
                record,
            )
            return record

    record = asyncio.run(run())
    assert record["stream_chunks"] == 2
    assert record["ttft_ms"] == pytest.approx(100)
    assert record["e2e_ms"] == pytest.approx(300)
    assert record["tpot_ms"] == pytest.approx(200 / 3)
    assert stream.closed and stream.exhausted


def test_one_special_token_has_ttft_but_no_tpot():
    values = [
        {"choices": [{"text": "", "token_ids": [1], "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        "[DONE]",
    ]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=Chunks(sse(values)))
            )
        ) as client:
            record = {"sent_at": performance.time.perf_counter()}
            await performance.stream_request(
                client,
                "http://test/v1/completions",
                {"prompt": [1], "max_tokens": 1},
                record,
            )
            return record

    record = asyncio.run(run())
    assert record["ttft_ms"] > 0 and record["tpot_ms"] is None


@pytest.mark.parametrize(
    "fault",
    [
        "missing-done",
        "missing-usage",
        "wrong-input",
        "wrong-output",
        "bool-count",
        "stream-error",
        "no-tokens",
    ],
)
def test_invalid_stream_cannot_supply_successful_metrics(fault):
    values = stream_values()
    if fault == "missing-done":
        values.pop()
    elif fault == "missing-usage":
        values.pop(-2)
    elif fault in {"wrong-input", "wrong-output", "bool-count"}:
        key = "prompt_tokens" if fault == "wrong-input" else "completion_tokens"
        values[-2]["usage"][key] = True if fault == "bool-count" else 99
    elif fault == "stream-error":
        values.insert(2, {"error": {"message": "backend failed"}})
    else:
        values = values[-3:]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=Chunks(sse(values)))
            )
        ) as client:
            record = {"sent_at": performance.time.perf_counter()}
            with pytest.raises(RuntimeError):
                await performance.stream_request(
                    client,
                    "http://test/v1/completions",
                    {"prompt": [1] * 8, "max_tokens": 4},
                    record,
                )
            assert record.get("status") != "passed"

    asyncio.run(run())


def test_concurrency_cap_and_queue_time_do_not_inflate_ttft(make_case):
    settings = asdict(
        make_case(
            "--test",
            "perf",
            "--perf-mode",
            "mixed",
            "--perf-input-lens",
            "8",
            "--perf-output-lens",
            "4",
            "--perf-concurrencies",
            "3",
            "--rps",
            "1000",
            "--perf-requests",
            "12",
        ).performance
    )
    point = performance.points(settings)[0]
    active = peak = 0

    class Slow(Chunks):
        def __init__(self, *args, **kwargs):
            nonlocal active, peak
            super().__init__(*args, **kwargs)
            active += 1
            peak = max(peak, active)

        async def aclose(self):
            nonlocal active
            if not self.closed:
                active -= 1
            await super().aclose()

    async def run():
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, stream=Slow(sse(stream_values()), delay=0.01))
        )
        async with httpx.AsyncClient(transport=transport) as client:
            records, raw, state = [], io.StringIO(), {}
            await performance.batch(
                client,
                "http://test/v1/completions",
                "model",
                [1, 2, 3],
                point,
                12,
                "measured",
                5,
                records,
                raw,
                state,
            )
            return records, raw, state

    records, raw, state = asyncio.run(run())
    assert active == 0 and peak == state["peak_concurrency"] == 3
    assert all(r["status"] == "passed" for r in records)
    assert max(r["client_queue_ms"] for r in records) > 50
    assert max(r["ttft_ms"] for r in records) < max(
        r["client_queue_ms"] for r in records
    )
    assert len(raw.getvalue().splitlines()) == 12


def test_total_request_timeout_survives_regular_stream_activity(make_case):
    point = performance.points(
        asdict(
            make_case(
                "--test",
                "perf",
                "--perf-input-lens",
                "8",
                "--perf-output-lens",
                "4",
                "--perf-concurrencies",
                "1",
            ).performance
        )
    )[0]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, stream=Chunks(sse(stream_values()), delay=0.03)
                )
            )
        ) as client:
            records, raw = [], io.StringIO()
            await performance.batch(
                client,
                "http://test/v1/completions",
                "model",
                [1, 2],
                point,
                2,
                "measured",
                0.07,
                records,
                raw,
                {},
            )
            return records

    records = asyncio.run(run())
    assert all(
        r["status"] == "failed" and "TimeoutError" in r["error"] for r in records
    )
    assert performance.aggregate(records)["metrics"]["ttft_ms"]["count"] == 0


def test_cancellation_preserves_raw_request_and_stops_inflight_work(make_case):
    point = performance.points(
        asdict(
            make_case(
                "--test",
                "perf",
                "--perf-input-lens",
                "8",
                "--perf-output-lens",
                "4",
                "--perf-concurrencies",
                "1",
                "--rps",
                ".1",
            ).performance
        )
    )[0]

    async def run():
        started = asyncio.Event()
        stream = Chunks(sse(stream_values()), delay=1)

        def handler(_request):
            started.set()
            return httpx.Response(200, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            records, raw = [], io.StringIO()
            task = asyncio.create_task(
                performance.batch(
                    client,
                    "http://test/v1/completions",
                    "model",
                    [1, 2],
                    point,
                    3,
                    "measured",
                    5,
                    records,
                    raw,
                    {},
                )
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(records) == 1 and records[0]["status"] == "cancelled"
            assert json.loads(raw.getvalue())["status"] == "cancelled"
            assert stream.closed

    asyncio.run(run())


def test_warmup_failure_keeps_actual_error_and_skips_measurements(
    make_case, tmp_path, monkeypatch
):
    config = make_case(
        "--test",
        "perf",
        "--perf-input-lens",
        "8",
        "--perf-output-lens",
        "4",
        "--perf-concurrencies",
        "1",
        "--perf-warmup",
        "2",
        "--perf-requests",
        "4",
    ).to_dict()
    requests = 0

    def handler(request):
        nonlocal requests
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        requests += 1
        if requests == 2:
            return httpx.Response(500, json={"error": "second warmup failed"})
        return httpx.Response(200, stream=Chunks(sse(stream_values())))

    native_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: native_client(transport=httpx.MockTransport(handler), **kw),
    )
    with pytest.raises(RuntimeError, match="warmup failed: HTTPStatusError"):
        asyncio.run(performance.evaluate(config, tmp_path))
    result = json.loads((tmp_path / "performance.json").read_text())
    point = result["points"][0]
    assert result["status"] == point["status"] == "failed"
    assert point["warmup"]["completed"] == point["warmup"]["failed"] == 1
    assert point["unsubmitted"] == 4 and point["completed"] == 0
    assert requests == 2 and "500" in result["errors"][0]


def test_cancelled_evaluation_keeps_partial_summary(make_case, tmp_path, monkeypatch):
    config = make_case(
        "--test",
        "perf",
        "--perf-input-lens",
        "8",
        "--perf-output-lens",
        "4",
        "--perf-concurrencies",
        "1",
        "--perf-warmup",
        "0",
        "--perf-requests",
        "3",
        "--rps",
        ".1",
    ).to_dict()
    native_client = httpx.AsyncClient
    stream = Chunks(sse(stream_values()), delay=1)

    async def run():
        started = asyncio.Event()

        def handler(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            started.set()
            return httpx.Response(200, stream=stream)

        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **kw: native_client(transport=httpx.MockTransport(handler), **kw),
        )
        task = asyncio.create_task(performance.evaluate(config, tmp_path))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    result = json.loads((tmp_path / "performance.json").read_text())
    point = result["points"][0]
    assert result["status"] == point["status"] == "cancelled"
    assert point["cancelled"] == 1 and point["unsubmitted"] == 2
    assert point["metrics"]["ttft_ms"]["count"] == 0 and stream.closed
    raw = tmp_path / "performance" / "requests.jsonl"
    assert json.loads(raw.read_text())["status"] == "cancelled"
