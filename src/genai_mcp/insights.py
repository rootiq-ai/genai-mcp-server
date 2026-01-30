"""Intelligent insights: cost optimization, root cause analysis, recommendations, SLA monitoring."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .prometheus import prom, get_value, get_values, parse_range_results
from .analytics import detect_anomalies, analyze_trend

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A recommendation with priority and impact."""
    category: str  # cost, quality, performance, security
    priority: str  # critical, high, medium, low
    title: str
    description: str
    impact: str
    action: str
    estimated_savings: float | None = None


@dataclass
class RootCauseResult:
    """Root cause analysis result."""
    issue: str
    timestamp: str
    severity: str
    probable_causes: list[dict[str, Any]]
    related_metrics: list[dict[str, Any]]
    recommendations: list[str]


@dataclass 
class SLAStatus:
    """SLA compliance status."""
    name: str
    target: float
    current: float
    is_compliant: bool
    margin: float
    trend: str
    time_to_breach: str | None


def get_cost_optimization_recommendations() -> list[Recommendation]:
    """
    Analyze cost metrics and provide optimization recommendations.
    
    Returns:
        List of cost optimization recommendations
    """
    recommendations = []
    
    # Get cost by model
    cost_by_model = get_values(prom.query('sum by (model) (genai_cost_dollars_total)'))
    requests_by_model = get_values(prom.query('sum by (model) (genai_requests_total)'))
    
    # Build model stats
    model_stats = {}
    for item in cost_by_model:
        model = item["metric"].get("model", "unknown")
        model_stats[model] = {"cost": item["value"], "requests": 0, "cost_per_request": 0}
    
    for item in requests_by_model:
        model = item["metric"].get("model", "unknown")
        if model in model_stats:
            model_stats[model]["requests"] = item["value"]
            if item["value"] > 0:
                model_stats[model]["cost_per_request"] = model_stats[model]["cost"] / item["value"]
    
    # Analyze and recommend
    if model_stats:
        # Find expensive models
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["cost_per_request"], reverse=True)
        
        if len(sorted_models) > 1:
            expensive = sorted_models[0]
            cheap = sorted_models[-1]
            
            if expensive[1]["cost_per_request"] > cheap[1]["cost_per_request"] * 2:
                potential_savings = (expensive[1]["cost_per_request"] - cheap[1]["cost_per_request"]) * expensive[1]["requests"]
                
                recommendations.append(Recommendation(
                    category="cost",
                    priority="high",
                    title=f"Consider replacing {expensive[0]} with {cheap[0]}",
                    description=f"{expensive[0]} costs ${expensive[1]['cost_per_request']:.6f}/request vs ${cheap[1]['cost_per_request']:.6f}/request for {cheap[0]}",
                    impact=f"Potential savings of ${potential_savings:.2f}",
                    action=f"Evaluate if {cheap[0]} can handle workloads currently on {expensive[0]}",
                    estimated_savings=potential_savings,
                ))
    
    # Check for high token usage
    input_tokens = get_value(prom.query('sum(genai_input_tokens_total)'))
    output_tokens = get_value(prom.query('sum(genai_output_tokens_total)'))
    
    if input_tokens > 0 and output_tokens / input_tokens > 3:
        recommendations.append(Recommendation(
            category="cost",
            priority="medium",
            title="High output-to-input token ratio",
            description=f"Output tokens ({int(output_tokens)}) are {output_tokens/input_tokens:.1f}x input tokens ({int(input_tokens)})",
            impact="May indicate verbose responses increasing costs",
            action="Review prompts to request more concise responses, use max_tokens limits",
            estimated_savings=None,
        ))
    
    # Check error rate (wasted spend)
    total_requests = get_value(prom.query('sum(genai_requests_total)'))
    total_errors = get_value(prom.query('sum(genai_errors_total)'))
    total_cost = get_value(prom.query('sum(genai_cost_dollars_total)'))
    
    if total_requests > 0:
        error_rate = total_errors / total_requests
        if error_rate > 0.05:  # >5% errors
            wasted_cost = total_cost * error_rate
            recommendations.append(Recommendation(
                category="cost",
                priority="high",
                title=f"High error rate ({error_rate*100:.1f}%) wasting budget",
                description=f"{int(total_errors)} failed requests out of {int(total_requests)}",
                impact=f"Approximately ${wasted_cost:.2f} wasted on failed requests",
                action="Investigate and fix error causes, implement retry logic with backoff",
                estimated_savings=wasted_cost,
            ))
    
    return recommendations


def get_quality_recommendations() -> list[Recommendation]:
    """Analyze quality metrics and provide recommendations."""
    recommendations = []
    
    # Get quality scores by application
    quality_metrics = ["groundedness", "relevance", "coherence", "fluency"]
    app_quality = {}
    
    for metric in quality_metrics:
        results = get_values(prom.query(f'genai_quality_{metric}'))
        for item in results:
            app = item["metric"].get("application", "unknown")
            if app not in app_quality:
                app_quality[app] = {}
            app_quality[app][metric] = item["value"]
    
    # Analyze each application
    for app, scores in app_quality.items():
        if not scores:
            continue
        
        avg_quality = sum(scores.values()) / len(scores)
        
        # Low overall quality
        if avg_quality < 0.7:
            recommendations.append(Recommendation(
                category="quality",
                priority="high",
                title=f"Low quality scores for {app}",
                description=f"Average quality: {avg_quality*100:.1f}%",
                impact="Poor user experience, potential hallucinations",
                action="Review prompts, add guardrails, consider fine-tuning or RAG",
            ))
        
        # Specific dimension issues
        if scores.get("groundedness", 1) < 0.7:
            recommendations.append(Recommendation(
                category="quality",
                priority="high",
                title=f"Low groundedness for {app}",
                description=f"Groundedness: {scores['groundedness']*100:.1f}%",
                impact="Responses may contain hallucinations or inaccurate information",
                action="Implement RAG, add fact-checking, provide more context in prompts",
            ))
        
        if scores.get("relevance", 1) < 0.7:
            recommendations.append(Recommendation(
                category="quality",
                priority="medium",
                title=f"Low relevance for {app}",
                description=f"Relevance: {scores['relevance']*100:.1f}%",
                impact="Responses may not address user queries effectively",
                action="Improve prompt engineering, add examples, refine system instructions",
            ))
    
    return recommendations


def get_performance_recommendations() -> list[Recommendation]:
    """Analyze performance metrics and provide recommendations."""
    recommendations = []
    
    # Get latency by model
    latency_p95 = get_values(prom.query('genai_latency_seconds{quantile="0.95"}'))
    ttft = get_values(prom.query('genai_ttft_seconds'))
    
    for item in latency_p95:
        model = item["metric"].get("model", "unknown")
        latency_ms = item["value"] * 1000
        
        if latency_ms > 5000:  # >5s p95
            recommendations.append(Recommendation(
                category="performance",
                priority="high",
                title=f"High latency for {model}",
                description=f"P95 latency: {latency_ms:.0f}ms",
                impact="Poor user experience, timeouts",
                action="Consider caching, use streaming, optimize prompts, or switch to faster model",
            ))
        elif latency_ms > 2000:  # >2s p95
            recommendations.append(Recommendation(
                category="performance",
                priority="medium",
                title=f"Moderate latency for {model}",
                description=f"P95 latency: {latency_ms:.0f}ms",
                impact="May affect user experience",
                action="Monitor trends, consider optimization if increasing",
            ))
    
    # TTFT analysis
    for item in ttft:
        model = item["metric"].get("model", "unknown")
        ttft_ms = item["value"] * 1000
        
        if ttft_ms > 1000:  # >1s TTFT
            recommendations.append(Recommendation(
                category="performance",
                priority="medium",
                title=f"Slow time-to-first-token for {model}",
                description=f"TTFT: {ttft_ms:.0f}ms",
                impact="Delayed streaming response start",
                action="Check network latency, consider edge deployment or different region",
            ))
    
    return recommendations


def get_security_recommendations() -> list[Recommendation]:
    """Analyze security metrics and provide recommendations."""
    recommendations = []
    
    # Get security events
    guardrail_triggers = get_values(prom.query('sum by (trigger_type) (genai_guardrail_triggers_total)'))
    security_events = get_values(prom.query('sum by (event_type) (genai_security_events_total)'))
    
    total_requests = get_value(prom.query('sum(genai_requests_total)'))
    
    for item in guardrail_triggers:
        trigger_type = item["metric"].get("trigger_type", "unknown")
        count = item["value"]
        
        if total_requests > 0:
            rate = count / total_requests * 100
            
            if trigger_type == "prompt_injection" and count > 0:
                recommendations.append(Recommendation(
                    category="security",
                    priority="critical" if rate > 1 else "high",
                    title=f"Prompt injection attempts detected",
                    description=f"{int(count)} attempts ({rate:.2f}% of requests)",
                    impact="Potential security breach, data leakage, or manipulation",
                    action="Review blocked prompts, strengthen input validation, implement additional guardrails",
                ))
            
            if trigger_type == "pii_detected" and count > 0:
                recommendations.append(Recommendation(
                    category="security",
                    priority="high",
                    title=f"PII detected in requests",
                    description=f"{int(count)} instances ({rate:.2f}% of requests)",
                    impact="Privacy compliance risk (GDPR, CCPA, etc.)",
                    action="Implement PII redaction, review data handling policies, audit affected requests",
                ))
            
            if trigger_type == "jailbreak" and count > 0:
                recommendations.append(Recommendation(
                    category="security",
                    priority="high",
                    title=f"Jailbreak attempts detected",
                    description=f"{int(count)} attempts ({rate:.2f}% of requests)",
                    impact="Model may bypass safety guidelines",
                    action="Strengthen system prompts, add output validation, implement content filtering",
                ))
    
    return recommendations


def analyze_root_cause(
    metric: str,
    threshold: float | None = None,
    lookback: str = "1h",
) -> RootCauseResult | None:
    """
    Perform root cause analysis when a metric exceeds threshold.
    
    Args:
        metric: Metric to analyze (e.g., 'genai_errors_total', 'genai_latency_seconds')
        threshold: Threshold value (auto-detected if None)
        lookback: Time range to analyze
    
    Returns:
        Root cause analysis result
    """
    # Detect anomalies first
    anomalies = detect_anomalies(metric, lookback=lookback)
    
    if not anomalies:
        return None
    
    # Get the most severe anomaly
    anomaly = max(anomalies, key=lambda x: abs(x.z_score))
    
    probable_causes = []
    related_metrics = []
    recommendations = []
    
    # Analyze based on metric type
    if "error" in metric.lower():
        # Check for correlated issues
        # High latency correlation
        latency = get_value(prom.query('avg(genai_latency_seconds{quantile="0.95"})'))
        if latency > 3:
            probable_causes.append({
                "cause": "High latency causing timeouts",
                "confidence": 0.8,
                "evidence": f"P95 latency is {latency*1000:.0f}ms",
            })
            related_metrics.append({"metric": "genai_latency_seconds", "value": latency})
            recommendations.append("Implement timeout handling and retry logic")
        
        # Check error breakdown
        error_breakdown = get_values(prom.query('sum by (error_type) (genai_errors_total)'))
        for item in error_breakdown:
            error_type = item["metric"].get("error_type", "unknown")
            count = item["value"]
            probable_causes.append({
                "cause": f"Error type: {error_type}",
                "confidence": 0.9,
                "evidence": f"{int(count)} occurrences",
            })
        
        # Model-specific issues
        errors_by_model = get_values(prom.query('sum by (model) (genai_errors_total)'))
        for item in errors_by_model:
            model = item["metric"].get("model", "unknown")
            count = item["value"]
            if count > 0:
                related_metrics.append({"metric": f"errors_{model}", "value": count})
    
    elif "latency" in metric.lower():
        # Check token counts (longer responses = higher latency)
        output_tokens = get_value(prom.query('avg(rate(genai_output_tokens_total[5m]))'))
        if output_tokens > 1000:
            probable_causes.append({
                "cause": "High output token count",
                "confidence": 0.7,
                "evidence": f"Avg output rate: {output_tokens:.0f} tokens/s",
            })
            recommendations.append("Consider limiting max_tokens in requests")
        
        # Check request rate
        request_rate = get_value(prom.query('rate(genai_requests_total[5m])'))
        if request_rate > 10:
            probable_causes.append({
                "cause": "High request rate may cause throttling",
                "confidence": 0.6,
                "evidence": f"Request rate: {request_rate:.1f}/s",
            })
            recommendations.append("Implement request queuing or rate limiting")
    
    elif "cost" in metric.lower():
        # Analyze cost drivers
        cost_by_model = get_values(prom.query('sum by (model) (rate(genai_cost_dollars_total[1h]))'))
        for item in cost_by_model:
            model = item["metric"].get("model", "unknown")
            cost_rate = item["value"] * 3600  # per hour
            if cost_rate > 0.1:
                probable_causes.append({
                    "cause": f"High cost from {model}",
                    "confidence": 0.9,
                    "evidence": f"${cost_rate:.2f}/hour",
                })
                recommendations.append(f"Evaluate if {model} can be replaced with cheaper alternative")
    
    return RootCauseResult(
        issue=anomaly.message,
        timestamp=datetime.now().isoformat(),
        severity=anomaly.severity,
        probable_causes=probable_causes,
        related_metrics=related_metrics,
        recommendations=recommendations,
    )


def check_sla_compliance(
    slas: list[dict[str, Any]] | None = None,
) -> list[SLAStatus]:
    """
    Check SLA compliance for defined SLAs.
    
    Args:
        slas: List of SLA definitions. If None, uses defaults.
    
    Returns:
        List of SLA compliance statuses
    """
    if slas is None:
        slas = [
            {"name": "Latency P95", "metric": 'avg(genai_latency_seconds{quantile="0.95"}) * 1000', "target": 3000, "unit": "ms", "comparison": "<="},
            {"name": "Error Rate", "metric": 'sum(genai_errors_total) / sum(genai_requests_total) * 100', "target": 1, "unit": "%", "comparison": "<="},
            {"name": "Quality Score", "metric": 'avg(genai_quality_groundedness)', "target": 0.8, "unit": "", "comparison": ">="},
            {"name": "Availability", "metric": '(1 - sum(genai_errors_total) / sum(genai_requests_total)) * 100', "target": 99, "unit": "%", "comparison": ">="},
        ]
    
    results = []
    
    for sla in slas:
        current = get_value(prom.query(sla["metric"]))
        target = sla["target"]
        comparison = sla.get("comparison", "<=")
        
        # Determine compliance
        if comparison == "<=":
            is_compliant = current <= target
            margin = target - current
        else:  # >=
            is_compliant = current >= target
            margin = current - target
        
        # Analyze trend
        trends = analyze_trend(sla["metric"], lookback="6h")
        trend = "stable"
        time_to_breach = None
        
        if trends:
            trend_result = trends[0]
            trend = trend_result.direction
            
            # Estimate time to breach if trending badly
            if not is_compliant:
                time_to_breach = "Already breached"
            elif comparison == "<=" and trend == "increasing":
                # How long until we hit target?
                if trend_result.slope > 0:
                    hours_to_breach = (target - current) / (trend_result.slope * 60)
                    if hours_to_breach > 0 and hours_to_breach < 168:  # Less than a week
                        time_to_breach = f"~{hours_to_breach:.1f} hours"
            elif comparison == ">=" and trend == "decreasing":
                if trend_result.slope < 0:
                    hours_to_breach = (current - target) / abs(trend_result.slope * 60)
                    if hours_to_breach > 0 and hours_to_breach < 168:
                        time_to_breach = f"~{hours_to_breach:.1f} hours"
        
        results.append(SLAStatus(
            name=sla["name"],
            target=target,
            current=round(current, 4),
            is_compliant=is_compliant,
            margin=round(margin, 4),
            trend=trend,
            time_to_breach=time_to_breach,
        ))
    
    return results


def get_all_recommendations() -> dict[str, list[Recommendation]]:
    """Get all recommendations across all categories."""
    return {
        "cost": get_cost_optimization_recommendations(),
        "quality": get_quality_recommendations(),
        "performance": get_performance_recommendations(),
        "security": get_security_recommendations(),
    }


def generate_health_report() -> dict[str, Any]:
    """Generate comprehensive health report."""
    # Get basic stats
    total_requests = get_value(prom.query('sum(genai_requests_total)'))
    total_cost = get_value(prom.query('sum(genai_cost_dollars_total)'))
    total_errors = get_value(prom.query('sum(genai_errors_total)'))
    avg_latency = get_value(prom.query('avg(genai_latency_seconds{quantile="0.5"}) * 1000'))
    avg_quality = get_value(prom.query('avg(genai_quality_groundedness)'))
    
    error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
    
    # Determine health status
    issues = []
    if error_rate > 5:
        issues.append("High error rate")
    if avg_latency > 3000:
        issues.append("High latency")
    if avg_quality < 0.7:
        issues.append("Low quality")
    
    if len(issues) >= 2:
        health = "critical"
    elif len(issues) == 1:
        health = "warning"
    else:
        health = "healthy"
    
    # Get recommendations
    all_recs = get_all_recommendations()
    critical_recs = []
    for category, recs in all_recs.items():
        for rec in recs:
            if rec.priority in ["critical", "high"]:
                critical_recs.append({"category": category, "title": rec.title})
    
    # Check SLAs
    sla_status = check_sla_compliance()
    sla_breaches = [s.name for s in sla_status if not s.is_compliant]
    
    # Detect anomalies
    anomalies = []
    for metric in ["genai_errors_total", "genai_latency_seconds", "genai_cost_dollars_total"]:
        detected = detect_anomalies(metric)
        anomalies.extend([{"metric": a.metric_name, "severity": a.severity, "message": a.message} for a in detected])
    
    return {
        "timestamp": datetime.now().isoformat(),
        "health_status": health,
        "summary": {
            "total_requests": int(total_requests),
            "total_cost_usd": round(total_cost, 4),
            "error_rate_percent": round(error_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_quality": round(avg_quality, 3),
        },
        "issues": issues,
        "sla_breaches": sla_breaches,
        "anomalies": anomalies[:5],  # Top 5
        "critical_recommendations": critical_recs[:5],  # Top 5
    }
