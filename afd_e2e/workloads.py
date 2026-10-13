"""Cancellable concurrent HTTP smoke requests, in a child process."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from pathlib import Path

from .routing import routes


def evaluate(config: dict, kind: str, output: Path) -> None:
    import httpx

    concurrency = min(10, config["concurrency"])
    count = concurrency
    v4 = config["model"] == "v4-flash"
    max_tokens = 32
    if v4:
        max_tokens = 256
    role = "baseline" if config["baseline"] else "attention"
    endpoints = routes(config)
    path = "/v1/" + (
        "chat/completions" if v4 else "completions"
    )
    check_answer = v4
    results = [
        {"index": index, "error": "Request not completed"} for index in range(count)
    ]

    async def run_requests():
        semaphore = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(
            timeout=config["request_timeout"], trust_env=False
        ) as client:

            async def request(item):
                async with semaphore:
                    index = item["index"]
                    route = endpoints[index % len(endpoints)]
                    item["route"] = route
                    prompt = f"Compute {12 + index} + 7. Reply with just the number."
                    item["prompt"] = prompt
                    item["started_at"] = time.monotonic()
                    body = {
                        "model": f"afd-e2e-{role}",
                        "temperature": 0.0,
                        "max_tokens": max_tokens,
                    }
                    if v4:
                        body.update(
                            {
                                "messages": [{"role": "user", "content": prompt}],
                                "chat_template_kwargs": {"thinking": False},
                            }
                        )
                    else:
                        body["prompt"] = f"Question: {prompt}\nAnswer:"
                    try:
                        response = await client.post(
                            route["url"] + path, json=body, headers=route["headers"]
                        )
                        item["status_code"] = response.status_code
                        item["response_body"] = response.text
                        response.raise_for_status()
                        data = response.json()
                        item["response"] = data
                        choices = data["choices"]
                        if len(choices) != 1:
                            raise RuntimeError("Expected one returned choice")
                        choice = choices[0]
                        text = choice["message"]["content"] if v4 else choice["text"]
                        if (
                            not isinstance(text, str)
                            or not text.strip()
                            or not choice.get("finish_reason")
                        ):
                            raise RuntimeError(
                                "Response is empty or has no finish reason"
                            )
                        if check_answer and (
                            choice["finish_reason"] != "stop"
                            or text.strip() != str(19 + index)
                        ):
                            raise RuntimeError(
                                f"Expected {19 + index}, got {text!r} ({choice['finish_reason']})"
                            )
                        del item["error"]
                    except (
                        httpx.HTTPError,
                        ValueError,
                        KeyError,
                        TypeError,
                        RuntimeError,
                    ) as exc:
                        item["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        item["finished_at"] = time.monotonic()

            await asyncio.gather(*(request(item) for item in results))

    try:
        asyncio.run(run_requests())
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "kind": kind,
                    "concurrency": concurrency,
                    "answer_checked": check_answer,
                    "requests": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    errors = [
        f"request {item['index']}: {item['error']}"
        for item in results
        if "error" in item
    ]
    if errors:
        raise RuntimeError("; ".join(errors))
    if (
        concurrency > 1
        and max(item["started_at"] for item in results)
        >= min(item["finished_at"] for item in results)
    ):
        raise RuntimeError("Smoke requests did not overlap")
    print(
        f"{kind}: {count}/{count} requests passed; answer_checked={check_answer}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("smoke",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def cancel(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    evaluate(json.loads(args.config.read_text()), args.kind, args.output)


if __name__ == "__main__":
    main()
