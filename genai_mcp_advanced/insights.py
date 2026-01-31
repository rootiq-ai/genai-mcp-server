"""Intelligent insights: recommendations, root cause, SLA monitoring."""
from dataclasses import dataclass
from datetime import datetime
from .prometheus import prom, get_value, get_values
from .analytics import detect_anomalies, analyze_trend


@dataclass
class Recommendation:
    category: str
    priority: str
    title: str
    description: str
    impact: str
    action: str
    estimated_savings: float | None = None


@dataclass
class RootCauseResult:
    issue: str
    timestamp: str
    severity: str
    probable_causes: list
    related_metrics: list
    recommendations: list


@dataclass
class SLAStatus:
    name: str
    target: float
    current: float
    is_compliant: bool
    margin: float
    trend: str
    time_to_breach: str | None


def get_cost_recommendations() -> list[Recommendation]:
    """Cost optimization recommendations."""
    recs = []
    cost_by_model = get_values(prom.query('sum by (model) (genai_cost_dollars_total)'))
    requests_by_model = get_values(prom.query('sum by (model) (genai_requests_total)'))
    
    model_stats = {}
    for item in cost_by_model:
        model = item["metric"].get("model", "unknown")
        model_stats[model] = {"cost": item["value"], "requests": 0}
    for item in requests_by_model:
        model = item["metric"].get("model", "unknown")
        if model in model_stats:
            model_stats[model]["requests"] = item["value"]
            if item["value"] > 0:
                model_stats[model]["cost_per_request"] = model_stats[model]["cost"] / item["value"]
    
    if len(model_stats) > 1:
        sorted_models = sorted([(k, v) for k, v in model_stats.items() if v.get("cost_per_request")], key=lambda x: x[1].get("cost_per_request", 0), reverse=True)
        if len(sorted_models) > 1:
            exp, cheap = sorted_models[0], sorted_models[-1]
            if exp[1].get("cost_per_request", 0) > cheap[1].get("cost_per_request", 0) * 2:
                savings = (exp[1]["cost_per_request"] - cheap[1]["cost_per_request"]) * exp[1]["requests"]
                recs.append(Recommendation(
                    category="cost", priority="high",
                    title=f"Consider replacing {exp[0]} with {cheap[0]}",
                    description=f"{exp[0]} costs ${exp[1]['cost_per_request']:.6f}/req vs ${cheap[1]['cost_per_request']:.6f}/req",
                    impact=f"Potential savings: ${savings:.2f}", action=f"Evaluate if {cheap[0]} can handle workloads",
                    estimated_savings=savings
                ))
    
    total_req = get_value(prom.query('sum(genai_requests_total)'))
    total_err = get_value(prom.query('sum(genai_errors_total)'))
    total_cost = get_value(prom.query('sum(genai_cost_dollars_total)'))
    if total_req > 0 and total_err / total_req > 0.05:
        wasted = total_cost * (total_err / total_req)
        recs.append(Recommendation(
            category="cost", priority="high",
            title=f"High error rate ({total_err/total_req*100:.1f}%) wasting budget",
            description=f"{int(total_err)} failed out of {int(total_req)} requests",
            impact=f"~${wasted:.2f} wasted", action="Fix errors, implement retry logic",
            estimated_savings=wasted
        ))
    return recs


def get_quality_recommendations() -> list[Recommendation]:
    """Quality improvement recommendations."""
    recs = []
    for metric in ["groundedness", "relevance", "coherence", "fluency"]:
        results = get_values(prom.query(f'genai_quality_{metric}'))
        for item in results:
            app = item["metric"].get("application", "unknown")
            val = item["value"]
            if val < 0.7:
                recs.append(Recommendation(
                    category="quality", priority="high" if val < 0.6 else "medium",
                    title=f"Low {metric} for {app}",
                    description=f"{metric.title()}: {val*100:.1f}%",
                    impact="Poor user experience", action="Improve prompts, add RAG, or fine-tune"
                ))
    return recs


def get_performance_recommendations() -> list[Recommendation]:
    """Performance recommendations."""
    recs = []
    latency = get_values(prom.query('genai_latency_seconds{quantile="0.95"}'))
    for item in latency:
        model = item["metric"].get("model", "unknown")
        lat_ms = item["value"] * 1000
        if lat_ms > 5000:
            recs.append(Recommendation(
                category="performance", priority="high",
                title=f"High latency for {model}",
                description=f"P95: {lat_ms:.0f}ms",
                impact="Poor UX, timeouts", action="Use caching, streaming, or faster model"
            ))
    return recs


def get_security_recommendations() -> list[Recommendation]:
    """Security recommendations."""
    recs = []
    triggers = get_values(prom.query('sum by (trigger_type) (genai_guardrail_triggers_total)'))
    total_req = get_value(prom.query('sum(genai_requests_total)'))
    
    for item in triggers:
        t_type = item["metric"].get("trigger_type", "unknown")
        count = item["value"]
        if count > 0 and total_req > 0:
            rate = count / total_req * 100
            priority = "critical" if t_type == "prompt_injection" and rate > 1 else "high"
            recs.append(Recommendation(
                category="security", priority=priority,
                title=f"{t_type.replace('_', ' ').title()} detected",
                description=f"{int(count)} events ({rate:.2f}% of requests)",
                impact="Security risk", action="Review and strengthen guardrails"
            ))
    return recs


def get_all_recommendations() -> dict:
    """Get all recommendations."""
    return {
        "cost": get_cost_recommendations(),
        "quality": get_quality_recommendations(),
        "performance": get_performance_recommendations(),
        "security": get_security_recommendations()
    }


def analyze_root_cause(metric: str = "genai_errors_total", lookback: str = "1h") -> RootCauseResult | None:
    """Root cause analysis."""
    anomalies = detect_anomalies(metric, lookback=lookback)
    if not anomalies:
        return None
    
    anomaly = max(anomalies, key=lambda x: abs(x.z_score))
    causes, related, recommendations = [], [], []
    
    if "error" in metric.lower():
        latency = get_value(prom.query('avg(genai_latency_seconds{quantile="0.95"})'))
        if latency > 3:
            causes.append({"cause": "High latency causing timeouts", "confidence": 0.8, "evidence": f"P95 latency: {latency*1000:.0f}ms"})
            recommendations.append("Implement timeout handling")
        
        errors = get_values(prom.query('sum by (error_type) (genai_errors_total)'))
        for item in errors:
            causes.append({"cause": f"Error type: {item['metric'].get('error_type', 'unknown')}", "confidence": 0.9, "evidence": f"{int(item['value'])} occurrences"})
    
    return RootCauseResult(
        issue=anomaly.message, timestamp=datetime.now().isoformat(),
        severity=anomaly.severity, probable_causes=causes,
        related_metrics=related, recommendations=recommendations
    )


def check_sla_compliance(slas: list | None = None) -> list[SLAStatus]:
    """Check SLA compliance."""
    if slas is None:
        slas = [
            {"name": "Latency P95", "metric": 'avg(genai_latency_seconds{quantile="0.95"}) * 1000', "target": 3000, "comparison": "<="},
            {"name": "Error Rate", "metric": 'sum(genai_errors_total) / sum(genai_requests_total) * 100', "target": 1, "comparison": "<="},
            {"name": "Quality", "metric": 'avg(genai_quality_groundedness)', "target": 0.8, "comparison": ">="},
        ]
    
    results = []
    for sla in slas:
        current = get_value(prom.query(sla["metric"]))
        target = sla["target"]
        comp = sla.get("comparison", "<=")
        
        is_compliant = current <= target if comp == "<=" else current >= target
        margin = target - current if comp == "<=" else current - target
        
        trends = analyze_trend(sla["metric"], lookback="6h")
        trend = trends[0].direction if trends else "stable"
        
        time_to_breach = None
        if not is_compliant:
            time_to_breach = "Breached"
        
        results.append(SLAStatus(
            name=sla["name"], target=target, current=round(current, 4),
            is_compliant=is_compliant, margin=round(margin, 4),
            trend=trend, time_to_breach=time_to_breach
        ))
    return results


def generate_health_report() -> dict:
    """Generate health report."""
    total_req = get_value(prom.query('sum(genai_requests_total)'))
    total_cost = get_value(prom.query('sum(genai_cost_dollars_total)'))
    total_err = get_value(prom.query('sum(genai_errors_total)'))
    avg_latency = get_value(prom.query('avg(genai_latency_seconds{quantile="0.5"}) * 1000'))
    avg_quality = get_value(prom.query('avg(genai_quality_groundedness)'))
    
    error_rate = (total_err / total_req * 100) if total_req > 0 else 0
    
    issues = []
    if error_rate > 5:
        issues.append("High error rate")
    if avg_latency > 3000:
        issues.append("High latency")
    if avg_quality < 0.7:
        issues.append("Low quality")
    
    health = "critical" if len(issues) >= 2 else "warning" if len(issues) == 1 else "healthy"
    
    all_recs = get_all_recommendations()
    critical_recs = []
    for cat, recs in all_recs.items():
        for r in recs:
            if r.priority in ["critical", "high"]:
                critical_recs.append({"category": cat, "title": r.title})
    
    slas = check_sla_compliance()
    breaches = [s.name for s in slas if not s.is_compliant]
    
    anomalies = []
    for m in ["genai_errors_total", "genai_latency_seconds"]:
        for a in detect_anomalies(m):
            anomalies.append({"metric": a.metric_name, "severity": a.severity, "message": a.message})
    
    return {
        "timestamp": datetime.now().isoformat(),
        "health_status": health,
        "summary": {
            "total_requests": int(total_req),
            "total_cost_usd": round(total_cost, 4),
            "error_rate_percent": round(error_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_quality": round(avg_quality, 3)
        },
        "issues": issues,
        "sla_breaches": breaches,
        "anomalies": anomalies[:5],
        "critical_recommendations": critical_recs[:5]
    }
