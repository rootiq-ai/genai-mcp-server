#!/usr/bin/env python3
"""
GenAI Observability MCP Server - Advanced Edition.

Features:
- Basic metrics (requests, cost, tokens, latency, quality, RAG, security)
- Anomaly detection with Z-score
- Trend analysis and forecasting
- Root cause analysis
- Cost optimization recommendations
- SLA compliance monitoring
- Correlation analysis
- Period comparison

Usage:
    uv run genai-mcp-advanced
    uv run fastmcp run server.py --transport sse --port 3001
"""
from dataclasses import asdict
from typing import Optional
from fastmcp import FastMCP

from . import tools
from . import analytics
from . import insights

mcp = FastMCP(
    name="genai-observability-advanced",
    instructions="""GenAI Observability MCP Server - Advanced Edition.

BASIC METRICS:
- get_summary, get_cost, get_tokens, get_latency, get_quality
- get_rag_metrics, get_ttft, get_security, get_errors, get_requests

ADVANCED ANALYTICS:
- detect_anomalies: Find unusual metric values using Z-score
- analyze_trends: Identify trends with linear regression
- forecast_metric: Predict future values
- find_correlations: Discover related metrics
- compare_periods: Compare time periods
- calculate_percentiles: Get p50/p90/p95/p99

INTELLIGENT INSIGHTS:
- get_recommendations: Cost, quality, performance, security suggestions
- analyze_root_cause: Identify why metrics are abnormal
- check_sla_compliance: Monitor SLA status
- generate_health_report: Full system health check
""",
)


# =============================================================================
# BASIC METRIC TOOLS
# =============================================================================

@mcp.tool()
def get_summary(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get comprehensive summary of all GenAI metrics including requests, cost, tokens, latency, quality, and security."""
    return tools.get_summary(application, model)


@mcp.tool()
def get_cost(application: Optional[str] = None, model: Optional[str] = None, breakdown: str = "model") -> dict:
    """Get cost breakdown in USD by model or application."""
    return tools.get_cost(application, model, breakdown)


@mcp.tool()
def get_tokens(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get token usage (input/output tokens)."""
    return tools.get_tokens(application, model)


@mcp.tool()
def get_latency(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get latency percentiles (p50, p95, p99) in milliseconds."""
    return tools.get_latency(application, model)


@mcp.tool()
def get_quality(application: Optional[str] = None) -> dict:
    """Get quality metrics: groundedness, relevance, coherence, fluency."""
    return tools.get_quality(application)


@mcp.tool()
def get_rag_metrics(application: Optional[str] = None) -> dict:
    """Get RAG metrics: precision, recall, F1 score."""
    return tools.get_rag_metrics(application)


@mcp.tool()
def get_ttft(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get Time to First Token (TTFT) in milliseconds."""
    return tools.get_ttft(application, model)


@mcp.tool()
def get_security(application: Optional[str] = None) -> dict:
    """Get security events: prompt injection, PII detection, jailbreak attempts, guardrail triggers."""
    return tools.get_security(application)


@mcp.tool()
def get_errors(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get error counts by type and total error rate."""
    return tools.get_errors(application, model)


@mcp.tool()
def get_requests(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get request counts by model."""
    return tools.get_requests(application, model)


@mcp.tool()
def get_applications() -> dict:
    """List all applications sending GenAI metrics."""
    return tools.get_applications()


@mcp.tool()
def get_models() -> dict:
    """List all LLM models in use."""
    return tools.get_models()


@mcp.tool()
def execute_query(promql: str) -> dict:
    """Execute a custom PromQL query against Prometheus."""
    return tools.execute_query(promql)


# =============================================================================
# ADVANCED ANALYTICS TOOLS
# =============================================================================

@mcp.tool()
def detect_anomalies(
    metric: str = "genai_errors_total",
    lookback: str = "1h",
    z_threshold: float = 2.5,
) -> dict:
    """
    Detect anomalies in a metric using Z-score analysis.
    
    Args:
        metric: Metric name (e.g., 'genai_errors_total', 'genai_latency_seconds', 'genai_cost_dollars_total')
        lookback: Time range for baseline (e.g., '1h', '6h', '24h')
        z_threshold: Z-score threshold for anomaly (default 2.5 = 99% confidence)
    
    Returns:
        List of detected anomalies with severity (low/medium/high/critical)
    """
    results = analytics.detect_anomalies(metric, lookback=lookback, z_threshold=z_threshold)
    return {
        "anomalies": [asdict(r) for r in results],
        "count": len(results),
        "metric": metric,
        "lookback": lookback,
    }


@mcp.tool()
def analyze_trends(
    metric: str = "genai_requests_total",
    lookback: str = "6h",
) -> dict:
    """
    Analyze metric trends using linear regression.
    
    Args:
        metric: Metric name
        lookback: Time range for analysis
    
    Returns:
        Trend direction, slope, R², change %, and forecasts
    """
    results = analytics.analyze_trend(metric, lookback=lookback)
    return {
        "trends": [asdict(r) for r in results],
        "metric": metric,
        "lookback": lookback,
    }


@mcp.tool()
def forecast_metric(
    metric: str = "genai_cost_dollars_total",
    horizon_hours: int = 24,
    lookback: str = "24h",
) -> dict:
    """
    Forecast future metric values with confidence intervals.
    
    Args:
        metric: Metric to forecast
        horizon_hours: Hours to forecast ahead (1-168)
        lookback: Historical data to use for forecasting
    
    Returns:
        Predicted values with 95% confidence intervals
    """
    return analytics.forecast_metric(metric, horizon_hours, lookback)


@mcp.tool()
def find_correlations(
    metrics: Optional[list[str]] = None,
    lookback: str = "1h",
    min_correlation: float = 0.7,
) -> dict:
    """
    Find correlations between metrics.
    
    Args:
        metrics: List of metrics to compare. Default: errors, latency, cost, requests
        lookback: Time range for analysis
        min_correlation: Minimum absolute correlation to report (0.0-1.0)
    
    Returns:
        Pairs of correlated metrics with correlation coefficient and p-value
    """
    if metrics is None:
        metrics = [
            "sum(genai_errors_total)",
            "avg(genai_latency_seconds)",
            "sum(genai_cost_dollars_total)",
            "sum(genai_requests_total)",
        ]
    
    results = analytics.find_correlations(metrics, lookback, min_correlation)
    return {
        "correlations": [asdict(r) for r in results],
        "count": len(results),
    }


@mcp.tool()
def compare_periods(
    metric: str,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
) -> dict:
    """
    Compare metric values between two time periods.
    
    Args:
        metric: Metric name or PromQL
        period_a_start: Start of first period (ISO format, e.g., '2024-01-01T00:00:00Z')
        period_a_end: End of first period
        period_b_start: Start of second period
        period_b_end: End of second period
    
    Returns:
        Statistical comparison including mean change, t-test results
    """
    return analytics.compare_periods(
        metric,
        (period_a_start, period_a_end),
        (period_b_start, period_b_end),
    )


@mcp.tool()
def calculate_percentiles(
    metric: str,
    percentiles: Optional[list[float]] = None,
    lookback: str = "1h",
) -> dict:
    """
    Calculate percentiles for a metric.
    
    Args:
        metric: Metric name
        percentiles: List of percentiles (default: 50, 90, 95, 99)
        lookback: Time range
    
    Returns:
        Percentile values, min, max, mean, std
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99]
    return analytics.calculate_percentiles(metric, percentiles, lookback)


# =============================================================================
# INTELLIGENT INSIGHTS TOOLS
# =============================================================================

@mcp.tool()
def get_recommendations(category: Optional[str] = None) -> dict:
    """
    Get intelligent recommendations for improving your GenAI system.
    
    Args:
        category: Filter by category: 'cost', 'quality', 'performance', 'security', or None for all
    
    Returns:
        Prioritized recommendations with impact and action items
    """
    all_recs = insights.get_all_recommendations()
    
    if category:
        recs = all_recs.get(category, [])
        return {
            "category": category,
            "recommendations": [asdict(r) for r in recs],
            "count": len(recs),
        }
    
    # Flatten and prioritize
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_flat = []
    for cat, recs in all_recs.items():
        for r in recs:
            all_flat.append((cat, r))
    
    all_flat.sort(key=lambda x: priority_order.get(x[1].priority, 4))
    
    return {
        "recommendations": [
            {"category": cat, **asdict(r)} for cat, r in all_flat
        ],
        "count": len(all_flat),
        "by_category": {cat: len(recs) for cat, recs in all_recs.items()},
    }


@mcp.tool()
def analyze_root_cause(
    metric: str = "genai_errors_total",
    lookback: str = "1h",
) -> dict:
    """
    Perform root cause analysis for metric anomalies.
    
    Args:
        metric: Metric showing issues (e.g., 'genai_errors_total', 'genai_latency_seconds')
        lookback: Time range to analyze
    
    Returns:
        Probable causes with confidence, related metrics, and recommendations
    """
    result = insights.analyze_root_cause(metric, lookback=lookback)
    if result:
        return asdict(result)
    return {"message": "No anomalies detected for root cause analysis"}


@mcp.tool()
def check_sla_compliance() -> dict:
    """
    Check SLA compliance for latency, error rate, quality, and availability.
    
    Returns:
        SLA status, current vs target, compliance status, trend, time to breach
    """
    results = insights.check_sla_compliance()
    return {
        "slas": [asdict(r) for r in results],
        "compliant_count": sum(1 for r in results if r.is_compliant),
        "total_count": len(results),
        "breaches": [r.name for r in results if not r.is_compliant],
    }


@mcp.tool()
def generate_health_report() -> dict:
    """
    Generate comprehensive health report for the GenAI system.
    
    Returns:
        Health status, summary metrics, issues, SLA breaches, anomalies, critical recommendations
    """
    return insights.generate_health_report()


# =============================================================================
# RESOURCES
# =============================================================================

@mcp.resource("genai://summary")
def resource_summary() -> str:
    """Current GenAI metrics summary."""
    import json
    return json.dumps(tools.get_summary(), indent=2)


@mcp.resource("genai://health")
def resource_health() -> str:
    """System health report."""
    import json
    return json.dumps(insights.generate_health_report(), indent=2)


@mcp.resource("genai://recommendations")
def resource_recommendations() -> str:
    """Current recommendations."""
    import json
    all_recs = insights.get_all_recommendations()
    return json.dumps({
        cat: [asdict(r) for r in recs] for cat, recs in all_recs.items()
    }, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
