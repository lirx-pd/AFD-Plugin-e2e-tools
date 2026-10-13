"""Capture optional Prometheus endpoints without depending on server internals."""

import urllib.request
from http.client import HTTPException
from pathlib import Path

from .routing import routes


def capture_metrics(config: dict, output_dir: Path, phase: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = {}
    urls = dict.fromkeys(route["url"] for route in routes(config))
    for instance, url in enumerate(urls):
        try:
            with opener.open(url + "/metrics", timeout=3) as response:
                if response.status != 200:
                    results[url] = {
                        "status": "unavailable",
                        "error": f"HTTP {response.status}",
                    }
                    continue
                body = response.read()
        except (OSError, HTTPException, ValueError) as error:
            results[url] = {"status": "unavailable", "error": str(error)}
            continue
        path = output_dir / "metrics" / f"{phase}-{instance}.prom"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        results[url] = {"status": "captured", "path": str(path.relative_to(output_dir))}
    return results
