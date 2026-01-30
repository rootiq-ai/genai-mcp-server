"""Advanced Prometheus client with caching, range queries, and metadata."""
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from cachetools import TTLCache

logger = logging.getLogger(__name__)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


@dataclass
class MetricResult:
    """Structured metric result."""
    metric: dict[str, str]
    value: float
    timestamp: float = 0
    
    @property
    def labels(self) -> dict[str, str]:
        return self.metric
    
    def get_label(self, name: str, default: str = "unknown") -> str:
        return self.metric.get(name, default)


@dataclass
class RangeResult:
    """Structured range query result."""
    metric: dict[str, str]
    values: list[tuple[float, float]]  # (timestamp, value)
    
    @property
    def timestamps(self) -> list[float]:
        return [v[0] for v in self.values]
    
    @property
    def data_points(self) -> list[float]:
        return [v[1] for v in self.values]
    
    def get_label(self, name: str, default: str = "unknown") -> str:
        return self.metric.get(name, default)


class PrometheusClient:
    """Advanced Prometheus client with caching and range queries."""

    def __init__(self, url: str | None = None, cache_ttl: int = 5, cache_size: int = 1000):
        self.url = (url or PROMETHEUS_URL).rstrip("/")
        self._cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._metadata_cache = TTLCache(maxsize=100, ttl=300)  # 5 min cache for metadata

    def _request(self, endpoint: str, params: dict | None = None, timeout: int = 30) -> dict:
        """Make HTTP request to Prometheus."""
        url = f"{self.url}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        
        cache_key = url
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
                self._cache[cache_key] = result
                return result
        except Exception as e:
            logger.error(f"Prometheus request error: {e}")
            return {"status": "error", "error": str(e)}

    def query(self, promql: str, time: float | None = None) -> dict[str, Any]:
        """Execute instant query."""
        params = {"query": promql}
        if time:
            params["time"] = time
        return self._request("/api/v1/query", params)

    def query_range(
        self,
        promql: str,
        start: datetime | str | float,
        end: datetime | str | float,
        step: str = "60s",
    ) -> dict[str, Any]:
        """Execute range query."""
        def to_timestamp(t):
            if isinstance(t, datetime):
                return t.timestamp()
            if isinstance(t, str):
                return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
            return t

        params = {
            "query": promql,
            "start": to_timestamp(start),
            "end": to_timestamp(end),
            "step": step,
        }
        return self._request("/api/v1/query_range", params, timeout=60)

    def query_range_relative(
        self,
        promql: str,
        duration: str = "1h",
        step: str = "60s",
    ) -> dict[str, Any]:
        """Execute range query with relative time (e.g., '1h', '24h', '7d')."""
        end = datetime.now()
        
        # Parse duration
        unit = duration[-1]
        value = int(duration[:-1])
        if unit == 'm':
            start = end - timedelta(minutes=value)
        elif unit == 'h':
            start = end - timedelta(hours=value)
        elif unit == 'd':
            start = end - timedelta(days=value)
        else:
            start = end - timedelta(hours=1)
        
        return self.query_range(promql, start, end, step)

    def get_label_values(self, label: str, metric: str | None = None) -> list[str]:
        """Get label values."""
        endpoint = f"/api/v1/label/{label}/values"
        params = {"match[]": metric} if metric else None
        result = self._request(endpoint, params)
        return result.get("data", [])

    def get_series(self, match: str | list[str], start: str = "1h") -> list[dict]:
        """Get time series metadata."""
        if isinstance(match, str):
            match = [match]
        
        end = datetime.now()
        start_dt = end - timedelta(hours=1)  # Default 1h
        
        params = {
            "match[]": match,
            "start": start_dt.timestamp(),
            "end": end.timestamp(),
        }
        result = self._request("/api/v1/series", params)
        return result.get("data", [])

    def get_metadata(self, metric: str | None = None) -> dict:
        """Get metric metadata."""
        cache_key = f"metadata:{metric}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]
        
        endpoint = "/api/v1/metadata"
        params = {"metric": metric} if metric else None
        result = self._request(endpoint, params)
        
        if result.get("status") == "success":
            self._metadata_cache[cache_key] = result.get("data", {})
        return result.get("data", {})

    def is_connected(self) -> bool:
        """Check if Prometheus is reachable."""
        try:
            result = self.query("up")
            return result.get("status") == "success"
        except Exception:
            return False

    def get_targets(self) -> list[dict]:
        """Get scrape targets status."""
        result = self._request("/api/v1/targets")
        return result.get("data", {}).get("activeTargets", [])

    def get_rules(self) -> dict:
        """Get alerting and recording rules."""
        result = self._request("/api/v1/rules")
        return result.get("data", {})

    def get_alerts(self) -> list[dict]:
        """Get active alerts."""
        result = self._request("/api/v1/alerts")
        return result.get("data", {}).get("alerts", [])


def parse_instant_results(resp: dict) -> list[MetricResult]:
    """Parse instant query results into structured objects."""
    results = []
    if resp.get("status") == "success":
        for item in resp.get("data", {}).get("result", []):
            ts, val = item.get("value", [0, 0])
            results.append(MetricResult(
                metric=item.get("metric", {}),
                value=float(val),
                timestamp=float(ts),
            ))
    return results


def parse_range_results(resp: dict) -> list[RangeResult]:
    """Parse range query results into structured objects."""
    results = []
    if resp.get("status") == "success":
        for item in resp.get("data", {}).get("result", []):
            values = [(float(ts), float(val)) for ts, val in item.get("values", [])]
            results.append(RangeResult(
                metric=item.get("metric", {}),
                values=values,
            ))
    return results


def get_value(resp: dict, default: float = 0) -> float:
    """Extract single value from response."""
    results = parse_instant_results(resp)
    return results[0].value if results else default


def get_values(resp: dict) -> list[dict]:
    """Extract values as dicts for compatibility."""
    return [{"metric": r.metric, "value": r.value} for r in parse_instant_results(resp)]


# Global client instance
prom = PrometheusClient()
