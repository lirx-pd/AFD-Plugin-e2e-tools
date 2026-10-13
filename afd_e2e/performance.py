"""Synthetic token-length workloads and client-observed streaming latency."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import random
import signal
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .reporting import write_json
from .routing import routes

SEED = 42
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms", "client_queue_ms", "scheduler_lag_ms")


def request_lengths(point: dict, count: int):
    rng = random.Random(SEED)
    for _ in range(count):
        yield (
            rng.choices(point["input_lens"], weights=point["weights"])[0],
            rng.choice(point["output_lens"]),
        )


def points(config: dict) -> list[dict]:
    workloads = []
    if config["mode"] in {"matrix", "both"}:
        workloads.extend(
            {"mode": "matrix", "input_lens": [i], "output_lens": [o], "weights": [1]}
            for i, o in itertools.product(config["input_lens"], config["output_lens"])
        )
    if config["mode"] in {"mixed", "both"}:
        workloads.append(
            {key: config[key] for key in ("input_lens", "output_lens", "weights")}
            | {"mode": "mixed"}
        )
    result = [
        {
            "id": f"point-{index:03d}",
            **workload,
            "phase": config["phase"],
            "concurrency": concurrency,
            "rps": rate,
            "arrival": config["arrival"],
            "requests": config["requests"],
        }
        for index, (workload, concurrency, rate) in enumerate(
            itertools.product(workloads, config["concurrencies"], config["rps"])
        )
    ]
    for point in result:
        counts = Counter(request_lengths(point, point["requests"]))
        point["planned_mix"] = [
            {"input_tokens": i, "output_tokens": o, "requests": counts[i, o],
             "ratio": counts[i, o] / point["requests"]}
            for i, o in itertools.product(point["input_lens"], point["output_lens"])
        ]
        missing_inputs = [
            i for i, weight in zip(point["input_lens"], point["weights"])
            if weight > 0 and not any(counts[i, o] for o in point["output_lens"])
        ]
        missing_outputs = [
            o for o in point["output_lens"]
            if not any(counts[i, o] for i in point["input_lens"])
        ]
        point["warnings"] = [
            f"No measured requests planned for {side} lengths {missing}; "
            "increase --perf-requests or adjust the length mix"
            for side, missing in (("input", missing_inputs), ("output", missing_outputs))
            if missing
        ]
    return result


def distribution(values: list[float]) -> dict:
    values = sorted(values)
    result = {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
    }
    for percentile in (50, 90, 95, 99):
        position = (len(values) - 1) * percentile / 100
        result[f"p{percentile}"] = (
            (
                values[math.floor(position)] * (1 - position % 1)
                + values[math.ceil(position)] * (position % 1)
            )
            if values
            else None
        )
    return result


def aggregate(records: list[dict]) -> dict:
    good = [record for record in records if record["status"] == "passed"]
    return {
        "completed": len(good),
        "failed": sum(record["status"] == "failed" for record in records),
        "cancelled": sum(record["status"] == "cancelled" for record in records),
        "metrics": {
            name: distribution(
                [
                    record[name]
                    for record in (
                        records
                        if name in {"client_queue_ms", "scheduler_lag_ms"}
                        else good
                    )
                    if record.get(name) is not None
                ]
            )
            for name in METRICS
        },
    }


def arrivals(count: int, rps: float | None, concurrency: int, mode: str) -> list[float]:
    rng = random.Random(SEED + 1)
    offsets = []
    current = 0.0
    for index in range(count):
        if rps is None:
            current = 0.0
        elif mode == "constant":
            current = index / rps
        elif mode == "burst":
            current = (index // concurrency) * concurrency / rps
        elif index:
            current += rng.expovariate(rps)
        offsets.append(current)
    return offsets


async def events(response):
    buffer = ""
    data = []
    async for block in response.aiter_text():
        buffer += block
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line and data:
                yield "\n".join(data)
                data.clear()
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
    if buffer.startswith("data:"):
        data.append(buffer[5:].removeprefix(" ").removesuffix("\r"))
    if data:
        yield "\n".join(data)


async def stream_request(client, url: str, body: dict, record: dict, headers: dict) -> None:
    first = last = None
    usage = None
    finished = None
    done = False
    chunks = 0
    async with client.stream("POST", url, json=body, headers=headers) as response:
        response.raise_for_status()
        async for event in events(response):
            if done:
                raise RuntimeError("Stream continued after [DONE]")
            if event == "[DONE]":
                done = True
                # Read through HTTP EOF so the connection can be reused.
                continue
            data = json.loads(event)
            if not isinstance(data, dict) or data.get("error"):
                raise RuntimeError(f"Invalid/error stream event: {data!r}")
            if data.get("usage") is not None:
                usage = data["usage"]
            if choices := data.get("choices"):
                if len(choices) != 1 or not isinstance(choices[0], dict):
                    raise RuntimeError("Expected one streamed choice")
                choice = choices[0]
                if choice.get("finish_reason"):
                    finished = choice["finish_reason"]
                # Token IDs distinguish generated special tokens from empty control chunks.
                emitted = (
                    bool(choice.get("token_ids"))
                    if choice.get("token_ids") is not None
                    else bool(choice.get("text"))
                )
                if emitted:
                    last = time.perf_counter()
                    if first is None:
                        first = last
                    chunks += 1
    if not done or finished not in {"stop", "length"} or first is None:
        raise RuntimeError(
            "Incomplete stream: need generated tokens, finish reason and [DONE]"
        )
    if not isinstance(usage, dict):
        raise RuntimeError("Stream has no token usage; cannot measure TPOT")
    for name, expected in (
        ("prompt_tokens", len(body["prompt"])),
        ("completion_tokens", body["max_tokens"]),
    ):
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RuntimeError(f"Expected {name}={expected}, got {value!r}")
    sent = record["sent_at"]
    count = usage["completion_tokens"]
    record.update(
        status="passed",
        input_tokens=usage["prompt_tokens"],
        output_tokens=count,
        stream_chunks=chunks,
        finish_reason=finished,
        ttft_ms=(first - sent) * 1000,
        e2e_ms=(last - sent) * 1000,
        tpot_ms=(last - first) * 1000 / (count - 1) if count > 1 else None,
    )


async def batch(
    client, endpoints, model, vocabulary, point, count, phase, timeout, records, raw, state
):
    import httpx

    semaphore = asyncio.Semaphore(point["concurrency"])
    lengths = iter(request_lengths(point, count))
    start = time.perf_counter()
    active = peak = 0
    # shortcut: exact percentiles keep all requests in memory, use disk aggregation for very large runs.
    tasks = []

    async def request(record):
        nonlocal active, peak
        try:
            async with semaphore:
                prompt_rng = random.Random(record["prompt_seed"])
                prompt = prompt_rng.choices(vocabulary, k=record["input_target"])
                body = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": record["output_target"],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "temperature": 0,
                    "ignore_eos": True,
                    "add_special_tokens": False,
                    "return_token_ids": True,
                }
                sent = time.perf_counter()
                record.update(
                    sent_at=sent,
                    sent_offset_s=sent - start,
                    client_queue_ms=(sent - start - record["arrival_offset_s"]) * 1000,
                )
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.wait_for(
                        stream_request(
                            client, record["route"]["url"] + "/v1/completions",
                            body, record, record["route"]["headers"],
                        ), timeout
                    )
                finally:
                    active -= 1
        except asyncio.CancelledError:
            record.update(status="cancelled", error="Request cancelled")
            raise
        except (
            httpx.HTTPError,
            asyncio.TimeoutError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
        ) as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            record.pop("sent_at", None)
            record["finished_offset_s"] = time.perf_counter() - start
            raw.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw.flush()

    try:
        for index, scheduled in enumerate(
            arrivals(count, point["rps"], point["concurrency"], point["arrival"])
        ):
            await asyncio.sleep(max(0, start + scheduled - time.perf_counter()))
            now = time.perf_counter() - start
            input_length, output_length = next(lengths)
            record = {
                "point": point["id"],
                "phase": phase,
                "performance_phase": point["phase"],
                "index": index,
                "route": endpoints[index % len(endpoints)],
                "input_target": input_length,
                "output_target": output_length,
                "prompt_seed": SEED + index + (1000000 if phase == "warmup" else 0),
                "scheduled_offset_s": scheduled,
                "arrival_offset_s": now,
                "scheduler_lag_ms": (now - scheduled) * 1000,
                "status": "pending",
            }
            records.append(record)
            tasks.append(asyncio.create_task(request(record)))
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        state.update(
            duration_seconds=time.perf_counter() - start, peak_concurrency=peak
        )


async def evaluate(config: dict, run_dir: Path) -> dict:
    import httpx

    settings = config["performance"]
    endpoints = routes(config)
    warmup_count = max(settings["warmup"], len(endpoints)) if settings["warmup"] else 0
    base = endpoints[0]["url"]
    url = base + "/v1/completions"
    model = "afd-e2e-baseline" if config["baseline"] else "afd-e2e-attention"
    result = {
        "status": "running",
        "endpoint": url,
        "routes": endpoints,
        "input_mode": "token_ids",
        "seed": SEED,
        "configuration": settings,
        "points": [],
        "errors": [],
    }
    directory = run_dir / "performance"
    directory.mkdir()
    try:
        capacity = max(settings["concurrencies"])
        async with httpx.AsyncClient(
            timeout=config["request_timeout"],
            trust_env=False,
            limits=httpx.Limits(
                max_connections=capacity, max_keepalive_connections=capacity
            ),
        ) as client:
            response = await asyncio.wait_for(
                client.post(
                    base + "/tokenize",
                    json={
                        "model": model,
                        "prompt": "A quiet river flows past the city. Researchers compare networks, memory, mathematics, language and science.",
                        "add_special_tokens": False,
                    },
                ),
                config["request_timeout"],
            )
            response.raise_for_status()
            vocabulary = response.json()["tokens"]
            if (
                not isinstance(vocabulary, list)
                or not vocabulary
                or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 0
                    for v in vocabulary
                )
            ):
                raise RuntimeError("/tokenize must return nonempty valid token IDs")
            result["seed_token_ids"] = vocabulary
            with (directory / "requests.jsonl").open("x", encoding="utf-8") as raw:
                planned = points(settings)
                for number, point in enumerate(planned, 1):
                    entry = {
                        **point,
                        "warmup_requests": warmup_count,
                        "status": "running",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                    result["points"].append(entry)
                    write_json(run_dir / "performance.json", result)
                    print(f"Performance {number}/{len(planned)}: {point}", flush=True)
                    records, warmup, state = [], [], {}
                    try:
                        await batch(
                            client,
                            endpoints,
                            model,
                            vocabulary,
                            {**point, "rps": None},
                            warmup_count,
                            "warmup",
                            config["request_timeout"],
                            warmup,
                            raw,
                            {},
                        )
                        failed = next(
                            (r for r in warmup if r["status"] != "passed"), None
                        )
                        if failed:
                            raise RuntimeError(
                                f"Performance warmup failed: {failed['error']}"
                            )
                        await batch(
                            client,
                            endpoints,
                            model,
                            vocabulary,
                            point,
                            point["requests"],
                            "measured",
                            config["request_timeout"],
                            records,
                            raw,
                            state,
                        )
                    except (asyncio.CancelledError, KeyboardInterrupt):
                        entry["status"] = "cancelled"
                        raise
                    finally:
                        entry.update(
                            aggregate(records),
                            **state,
                            warmup=aggregate(warmup),
                            unsubmitted=point["requests"] - len(records),
                        )
                        if entry["status"] == "running":
                            entry["status"] = (
                                "passed"
                                if entry["completed"] == point["requests"]
                                else "failed"
                            )
                        duration = state.get("duration_seconds", 0)
                        good = [
                            record for record in records if record["status"] == "passed"
                        ]
                        entry["achieved_rps"] = (
                            len(good) / duration if duration else None
                        )
                        for side in ("input", "output"):
                            entry[f"{side}_tokens_per_second"] = (
                                sum(record[f"{side}_tokens"] for record in good)
                                / duration
                                if duration
                                else None
                            )
                        sends = sorted(
                            record["sent_offset_s"]
                            for record in records
                            if "sent_offset_s" in record
                        )
                        entry["dispatch_rps"] = (
                            (len(sends) - 1) / (sends[-1] - sends[0])
                            if len(sends) > 1 and sends[-1] > sends[0]
                            else None
                        )
                        groups = sorted(
                            {(r["input_target"], r["output_target"]) for r in records}
                        )
                        entry["by_length"] = [
                            {
                                "input_tokens": i,
                                "output_tokens": o,
                                **aggregate(
                                    [
                                        r
                                        for r in records
                                        if (r["input_target"], r["output_target"])
                                        == (i, o)
                                    ]
                                ),
                            }
                            for i, o in groups
                        ]
                        entry["errors"] = [
                            {"index": r["index"], "error": r["error"]}
                            for r in records
                            if r.get("error")
                        ][:20]
                        write_json(run_dir / "performance.json", result)
                    print(
                        f"{point['id']}: {entry['completed']}/{point['requests']} passed; TTFT={entry['metrics']['ttft_ms']}; TPOT={entry['metrics']['tpot_ms']}",
                        flush=True,
                    )
        result["status"] = (
            "passed"
            if all(point["status"] == "passed" for point in result["points"])
            else "failed"
        )
    except BaseException as exc:
        result["status"] = (
            "cancelled"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed"
        )
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "performance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    def cancel(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    result = asyncio.run(
        evaluate(json.loads(args.config.read_text()), args.config.parent)
    )
    if result["status"] != "passed":
        raise RuntimeError(
            "Performance requests failed; see performance.json and performance/requests.jsonl"
        )


if __name__ == "__main__":
    main()
