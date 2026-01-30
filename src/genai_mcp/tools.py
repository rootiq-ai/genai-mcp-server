"""Basic GenAI Observability metric tools."""
from typing import Any
from .prometheus import prom, get_value, get_values


def get_summary(application: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Get comprehensive summary of all GenAI metrics."""
    f = []
    if application:
        f.append(f'application="{application}"')
    if model:
        f.append(f'model="{model}"')
    fs = ",".join(f) if f else 'application=~".+"'

    s = {}
    s["total_requests"] = int(get_value(prom.query(f"sum(genai_requests_total{{{fs}}})")))
    s["total_cost_usd"] = round(get_value(prom.query(f"sum(genai_cost_dollars_total{{{fs}}})")), 6)
    s["total_input_tokens"] = int(get_value(prom.query(f"sum(genai_input_tokens_total{{{fs}}})")))
    s["total_output_tokens"] = int(get_value(prom.query(f"sum(genai_output_tokens_total{{{fs}}})")))
    s["total_tokens"] = s["total_input_tokens"] + s["total_output_tokens"]
    s["total_errors"] = int(get_value(prom.query(f"sum(genai_errors_total{{{fs}}}) or vector(0)")))
    s["avg_latency_ms"] = round(get_value(prom.query(f'avg(genai_latency_seconds{{{fs}, quantile="0.5"}}) * 1000')), 2)
    s["p95_latency_ms"] = round(get_value(prom.query(f'avg(genai_latency_seconds{{{fs}, quantile="0.95"}}) * 1000')), 2)
    s["avg_groundedness"] = round(get_value(prom.query("avg(genai_quality_groundedness)")), 3)
    s["avg_relevance"] = round(get_value(prom.query("avg(genai_quality_relevance)")), 3)
    s["avg_coherence"] = round(get_value(prom.query("avg(genai_quality_coherence)")), 3)
    s["avg_fluency"] = round(get_value(prom.query("avg(genai_quality_fluency)")), 3)
    s["avg_rag_precision"] = round(get_value(prom.query("avg(genai_rag_precision)")), 3)
    s["avg_rag_recall"] = round(get_value(prom.query("avg(genai_rag_recall)")), 3)
    s["security_events"] = int(get_value(prom.query("sum(genai_guardrail_triggers_total) or vector(0)")))
    s["avg_ttft_ms"] = round(get_value(prom.query(f"avg(genai_ttft_seconds{{{fs}}}) * 1000")), 2)
    quality = [s[f"avg_{m}"] for m in ["groundedness", "relevance", "coherence", "fluency"] if s.get(f"avg_{m}")]
    s["avg_quality"] = round(sum(quality) / len(quality), 3) if quality else 0
    s["error_rate_percent"] = round((s["total_errors"] / s["total_requests"] * 100) if s["total_requests"] > 0 else 0, 2)
    return {"summary": s, "filters": {"application": application, "model": model}}


def get_cost(application: str | None = None, model: str | None = None, breakdown: str = "model") -> dict:
    """Get cost breakdown."""
    f = []
    if application:
        f.append(f'application="{application}"')
    if model:
        f.append(f'model="{model}"')
    fs = ",".join(f) if f else 'application=~".+"'
    results, total = [], 0
    for item in get_values(prom.query(f"sum by ({breakdown}) (genai_cost_dollars_total{{{fs}}})")):
        cost = round(item["value"], 6)
        total += cost
        results.append({breakdown: item["metric"].get(breakdown, "unknown"), "cost_usd": cost})
    return {"breakdown": results, "total_cost_usd": round(total, 6)}


def get_tokens(application: str | None = None, model: str | None = None) -> dict:
    """Get token usage."""
    f = []
    if application:
        f.append(f'application="{application}"')
    if model:
        f.append(f'model="{model}"')
    fs = ",".join(f) if f else 'application=~".+"'
    inp = int(get_value(prom.query(f"sum(genai_input_tokens_total{{{fs}}})")))
    out = int(get_value(prom.query(f"sum(genai_output_tokens_total{{{fs}}})")))
    return {"total_input": inp, "total_output": out, "total": inp + out}


def get_latency(application: str | None = None, model: str | None = None) -> dict:
    """Get latency percentiles."""
    f = []
    if application:
        f.append(f'application="{application}"')
    if model:
        f.append(f'model="{model}"')
    fs = ",".join(f) if f else 'application=~".+"'
    results = {}
    for q in ["0.5", "0.95", "0.99"]:
        for item in get_values(prom.query(f'genai_latency_seconds{{{fs}, quantile="{q}"}}')):
            key = item["metric"].get("model", "unknown")
            if key not in results:
                results[key] = {"model": key}
            results[key][f"p{q.replace('.', '')}"] = round(item["value"] * 1000, 2)
    return {"latency_ms": list(results.values())}


def get_quality(application: str | None = None) -> dict:
    """Get quality metrics."""
    af = f'application="{application}"' if application else 'application=~".+"'
    results = {}
    for m in ["groundedness", "relevance", "coherence", "fluency"]:
        for item in get_values(prom.query(f"genai_quality_{m}{{{af}}}")):
            app = item["metric"].get("application", "unknown")
            if app not in results:
                results[app] = {"application": app}
            results[app][m] = round(item["value"], 3)
    for scores in results.values():
        vals = [v for k, v in scores.items() if k != "application" and isinstance(v, float)]
        scores["overall"] = round(sum(vals) / len(vals), 3) if vals else 0
    return {"quality": list(results.values())}


def get_rag_metrics(application: str | None = None) -> dict:
    """Get RAG metrics."""
    af = f'application="{application}"' if application else 'application=~".+"'
    results = {}
    for m in ["precision", "recall"]:
        for item in get_values(prom.query(f"genai_rag_{m}{{{af}}}")):
            app = item["metric"].get("application", "unknown")
            if app not in results:
                results[app] = {"application": app}
            results[app][m] = round(item["value"], 3)
    for scores in results.values():
        p, r = scores.get("precision", 0), scores.get("recall", 0)
        scores["f1_score"] = round(2 * p * r / (p + r), 3) if (p + r) > 0 else 0
    return {"rag_metrics": list(results.values())}


def get_ttft(application: str | None = None, model: str | None = None) -> dict:
    """Get Time to First Token."""
    f = []
    if application:
        f.append(f'application="{application}"')
    if model:
        f.append(f'model="{model}"')
    fs = ",".join(f) if f else 'application=~".+"'
    results = []
    for item in get_values(prom.query(f"genai_ttft_seconds{{{fs}}}")):
        results.append({"model": item["metric"].get("model", "unknown"), "ttft_ms": round(item["value"] * 1000, 2)})
    avg = sum(r["ttft_ms"] for r in results) / len(results) if results else 0
    return {"ttft": results, "avg_ttft_ms": round(avg, 2)}


def get_security(application: str | None = None) -> dict:
    """Get security metrics."""
    af = f'application="{application}"' if application else 'application=~".+"'
    triggers = [{"type": item["metric"].get("trigger_type", "unknown"), "count": int(item["value"])} 
                for item in get_values(prom.query(f"sum by (trigger_type) (genai_guardrail_triggers_total{{{af}}})"))]
    events = [{"type": item["metric"].get("event_type", "unknown"), "count": int(item["value"])} 
              for item in get_values(prom.query(f"sum by (event_type) (genai_security_events_total{{{af}}})"))]
    return {"guardrail_triggers": triggers, "security_events": events, "total_events": sum(t["count"] for t in triggers) + sum(e["count"] for e in events)}


def get_errors(application: str | None = None, model: str | None = None) -> dict:
    """Get error metrics."""
    f = []
    if application:
        f.append(f'application="{application}"')
    if model:
        f.append(f'model="{model}"')
    fs = ",".join(f) if f else 'application=~".+"'
    errors = [{"error_type": item["metric"].get("error_type", "unknown"), "count": int(item["value"])} 
              for item in get_values(prom.query(f"sum by (error_type) (genai_errors_total{{{fs}}})"))]
    return {"errors": errors, "total_errors": sum(e["count"] for e in errors)}


def get_requests(application: str | None = None, model: str | None = None) -> dict:
    """Get request counts."""
    f = []
    if application:
        f.append(f'application="{application}"')
    if model:
        f.append(f'model="{model}"')
    fs = ",".join(f) if f else 'application=~".+"'
    requests = [{"model": item["metric"].get("model", "unknown"), "count": int(item["value"])} 
                for item in get_values(prom.query(f"sum by (model) (genai_requests_total{{{fs}}})"))]
    return {"requests": requests, "total": sum(r["count"] for r in requests)}


def get_applications() -> dict:
    """Get list of applications."""
    return {"applications": prom.get_label_values("application", "genai_requests_total")}


def get_models() -> dict:
    """Get list of models."""
    return {"models": prom.get_label_values("model", "genai_requests_total")}


def execute_query(promql: str) -> dict:
    """Execute custom PromQL query."""
    resp = prom.query(promql)
    if resp.get("status") == "success":
        return {"status": "success", "results": get_values(resp)}
    return {"status": "error", "error": resp.get("error", "Query failed")}
