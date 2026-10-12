"""Map native DP domains to their HTTP instance and local DP rank."""

from .config import baseline_layouts


def routes(config: dict) -> list[dict]:
    host = "127.0.0.1" if config["api_host"] == "0.0.0.0" else config["api_host"]
    if not config["baseline"]:
        return [{"url": f"http://{host}:{config['api_port']}", "headers": {}}]
    result = []
    for instance, (dp, _) in enumerate(baseline_layouts(config["topology"], config["model"])):
        for rank in range(dp):
            result.append({
                "domain": len(result),
                "instance": instance,
                "dp_rank": rank,
                "url": f"http://{host}:{config['api_port'] + instance}",
                "headers": {"X-data-parallel-rank": str(rank)},
            })
    return result
