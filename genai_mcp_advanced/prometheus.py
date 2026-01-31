"""Prometheus client with caching."""
import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from cachetools import TTLCache

logger = logging.getLogger(__name__)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


@dataclass
class RangeResult:
    metric: dict[str, str]
    values: list[tuple[float, float]]
    
    @property
    def data_points(self) -> list[float]:
        return [v[1] for v in self.values]


class PrometheusClient:
    def __init__(self, url: str | None = None):
        self.url = (url or PROMETHEUS_URL).rstrip("/")
        self._cache = TTLCache(maxsize=1000, ttl=5)

    def _request(self, endpoint: str, params: dict | None = None, timeout: int = 30) -> dict:
        url = f"{self.url}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        if url in self._cache:
            return self._cache[url]
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
                self._cache[url] = result
                return result
        except Exception as e:
            logger.error(f"Prometheus error: {e}")
            return {"status": "error", "error": str(e)}

    def query(self, promql: str) -> dict[str, Any]:
        return self._request("/api/v1/query", {"query": promql})

    def query_range_relative(self, promql: str, duration: str = "1h", step: str = "60s") -> dict[str, Any]:
        end = datetime.now()
        unit, value = duration[-1], int(duration[:-1])
        delta = {"m": timedelta(minutes=value), "h": timedelta(hours=value), "d": timedelta(days=value)}.get(unit, timedelta(hours=1))
        start = end - delta
        params = {"query": promql, "start": start.timestamp(), "end": end.timestamp(), "step": step}
        return self._request("/api/v1/query_range", params, timeout=60)

    def get_label_values(self, label: str, metric: str | None = None) -> list[str]:
        endpoint = f"/api/v1/label/{label}/values"
        params = {"match[]": metric} if metric else None
        return self._request(endpoint, params).get("data", [])

    def is_connected(self) -> bool:
        try:
            return self.query("up").get("status") == "success"
        except:
            return False


def parse_range_results(resp: dict) -> list[RangeResult]:
    results = []
    if resp.get("status") == "success":
        for item in resp.get("data", {}).get("result", []):
            values = [(float(ts), float(val)) for ts, val in item.get("values", [])]
            results.append(RangeResult(metric=item.get("metric", {}), values=values))
    return results


def get_value(resp: dict, default: float = 0) -> float:
    if resp.get("status") == "success":
        results = resp.get("data", {}).get("result", [])
        if results:
            return float(results[0].get("value", [0, default])[1])
    return default


def get_values(resp: dict) -> list[dict]:
    results = []
    if resp.get("status") == "success":
        for item in resp.get("data", {}).get("result", []):
            results.append({"metric": item.get("metric", {}), "value": float(item.get("value", [0, 0])[1])})
    return results


prom = PrometheusClient()
