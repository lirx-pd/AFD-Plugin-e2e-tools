"""CPU subprocess fixture. It simulates protocols, never NPU inference."""

import json
import os
import re
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def argument(name):
    return sys.argv[sys.argv.index(name) + 1]


def service():
    role = os.environ["AFD_E2E_PROCESS_ROLE"]
    dp = int(argument("--data-parallel-size"))
    tp = int(argument("--tensor-parallel-size"))
    config = json.loads(argument("--additional-config"))
    activity = Path(os.environ["AFD_E2E_FAKE_ACTIVITY"])
    mrv = os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    layered = os.environ["AFD_E2E_LAYERED_EXPECTED"] == "1"
    name = (
        "AFDNPUFFNModelRunner"
        if role == "ffn"
        else "AFDNPUAttentionModelRunnerV2"
        if mrv
        else "AFDNPUAttentionModelRunner"
    )

    def emit(kind, **values):
        for d in range(dp):
            for t in range(tp):
                event = {
                    "time": time.time(),
                    "run_id": os.environ["AFD_E2E_RUN_ID"],
                    "role": role,
                    "kind": kind,
                    "dp_rank": d,
                    "tp_rank": t,
                    "class": name,
                    **values,
                }
                sys.stdout.write("AFD_E2E_EVENT " + json.dumps(event) + "\n")
                sys.stdout.flush()

    def execution():
        if role == "ffn" and os.environ.get("AFD_E2E_FAKE_LIVE_ERROR"):
            message = os.environ["AFD_E2E_FAKE_LIVE_ERROR"]
            print(
                "RuntimeError: simulated inference failure"
                if message == "1"
                else message,
                flush=True,
            )
        split = "--enable-dbo" in sys.argv or config["afd"][
            "connector_extra_config"
        ].get("async_moe_ubatching", False)
        emit(
            "execution",
            phase="live",
            mode="replay" if "--compilation-config" in sys.argv else "eager",
            stages=2 if split else 0,
        )
        if role == "attention" and mrv and split:
            emit(
                "native_ubatches",
                phase="live",
                executor="NPUUBatchRunner",
                capacity=[4, 4],
                real=[4, 4],
            )
        if layered:
            emit("layered", phase="live")

    emit("runner", mrv_flag=int(mrv), layered=layered)
    if role == "ffn":

        def monitor():
            last = None
            while True:
                stamp = activity.stat().st_mtime_ns if activity.exists() else None
                if stamp is not None and stamp != last:
                    execution()
                last = stamp
                time.sleep(0.01)

        threading.Thread(target=monitor, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, value):
            encoded = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.send({"data": [{"id": argument("--served-model-name")}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.send({"tokens": list(range(100, 125))})
                return
            activity.touch()
            execution()
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def chunk(value):
                    self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
                    self.wfile.flush()

                chunk(
                    {"choices": [{"text": "", "token_ids": [], "finish_reason": None}]}
                )
                count = body["max_tokens"]
                for index in range(0, count, 2):
                    time.sleep(0.01)
                    chunk(
                        {
                            "choices": [
                                {
                                    "text": "x",
                                    "token_ids": list(
                                        range(index, min(index + 2, count))
                                    ),
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                chunk(
                    {
                        "choices": [
                            {"text": "", "token_ids": [], "finish_reason": "length"}
                        ]
                    }
                )
                time.sleep(0.02)
                if not os.environ.get("AFD_E2E_FAKE_NO_USAGE"):
                    chunk(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": len(body["prompt"]),
                                "completion_tokens": count,
                            },
                        }
                    )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            time.sleep(0.1)
            text = body.get("prompt") or body["messages"][0]["content"]
            number = int(re.search(r"Compute (\d+) \+ 7", text)[1]) + 7
            if os.environ.get("AFD_E2E_FAKE_WRONG_ANSWER"):
                number += 1
            self.send(
                {
                    "choices": [
                        {
                            "text": str(number),
                            "message": {"content": str(number)},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", int(argument("--port"))), Handler)

    def stop(*_args):
        if role == "ffn" and os.environ.get("AFD_E2E_FAKE_SHUTDOWN_ERROR"):
            print(os.environ["AFD_E2E_FAKE_SHUTDOWN_ERROR"], flush=True)
        sys.exit(0)

    signal.signal(
        signal.SIGTERM,
        signal.SIG_IGN
        if role == "ffn" and os.environ.get("AFD_E2E_FAKE_IGNORE_TERM")
        else stop,
    )
    server.serve_forever()


if __name__ == "__main__":
    if sys.argv[1] == "service":
        service()
    elif sys.argv[1] == "preflight":
        print('AFD_E2E_PROBE {"simulated": true}', flush=True)
    elif sys.argv[1] == "evaluation":
        model_args = dict(
            part.split("=", 1) for part in argument("--model_args").split(",")
        )
        body = json.dumps({"prompt": "Compute 12 + 7"}).encode()
        request = urllib.request.Request(
            model_args["base_url"],
            data=body,
            headers={"Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=5) as response:
            assert response.status == 200
        output = Path(argument("--output_path"))
        output.mkdir()
        result = {
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": float(
                        os.environ.get("AFD_E2E_FAKE_SCORE", ".5")
                    )
                }
            },
            "n-samples": {"gsm8k": {"effective": int(argument("--limit"))}},
        }
        (output / "results_simulated.json").write_text(json.dumps(result))
