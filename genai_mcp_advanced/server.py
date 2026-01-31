"""FastMCP Server for GenAI Observability."""
from dataclasses import asdict
from typing import Optional
from fastmcp import FastMCP
from . import tools, analytics, insights

mcp = FastMCP(name="genai-observability-advanced")


@mcp.tool()
def get_summary(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get comprehensive metrics summary."""
    return tools.get_summary(application, model)


@mcp.tool()
def get_cost(application: Optional[str] = None, model: Optional[str] = None, breakdown: str = "model") -> dict:
    """Get cost breakdown."""
    return tools.get_cost(application, model, breakdown)


@mcp.tool()
def get_tokens(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get token usage."""
    return tools.get_tokens(application, model)


@mcp.tool()
def get_latency(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get latency percentiles."""
    return tools.get_latency(application, model)


@mcp.tool()
def get_quality(application: Optional[str] = None) -> dict:
    """Get quality metrics."""
    return tools.get_quality(application)


@mcp.tool()
def get_rag_metrics(application: Optional[str] = None) -> dict:
    """Get RAG metrics."""
    return tools.get_rag_metrics(application)


@mcp.tool()
def get_ttft(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get TTFT metrics."""
    return tools.get_ttft(application, model)


@mcp.tool()
def get_security(application: Optional[str] = None) -> dict:
    """Get security events."""
    return tools.get_security(application)


@mcp.tool()
def get_errors(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get error metrics."""
    return tools.get_errors(application, model)


@mcp.tool()
def get_requests(application: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Get request counts."""
    return tools.get_requests(application, model)


@mcp.tool()
def get_applications() -> dict:
    """List applications."""
    return tools.get_applications()


@mcp.tool()
def get_models() -> dict:
    """List models."""
    return tools.get_models()


@mcp.tool()
def execute_query(promql: str) -> dict:
    """Execute PromQL query."""
    return tools.execute_query(promql)


@mcp.tool()
def detect_anomalies(metric: str = "genai_errors_total", lookback: str = "1h", z_threshold: float = 2.5) -> dict:
    """Detect anomalies using Z-score."""
    results = analytics.detect_anomalies(metric, lookback=lookback, z_threshold=z_threshold)
    return {"anomalies": [asdict(r) for r in results], "count": len(results)}


@mcp.tool()
def analyze_trends(metric: str = "genai_requests_total", lookback: str = "6h") -> dict:
    """Analyze metric trends."""
    results = analytics.analyze_trend(metric, lookback=lookback)
    return {"trends": [asdict(r) for r in results]}


@mcp.tool()
def forecast_metric(metric: str = "genai_cost_dollars_total", horizon_hours: int = 24, lookback: str = "24h") -> dict:
    """Forecast future values."""
    return analytics.forecast_metric(metric, horizon_hours, lookback)


@mcp.tool()
def find_correlations(lookback: str = "1h", min_correlation: float = 0.7) -> dict:
    """Find metric correlations."""
    results = analytics.find_correlations(lookback=lookback, min_correlation=min_correlation)
    return {"correlations": [asdict(r) for r in results]}


@mcp.tool()
def calculate_percentiles(metric: str, lookback: str = "1h") -> dict:
    """Calculate percentiles."""
    return analytics.calculate_percentiles(metric, lookback=lookback)


@mcp.tool()
def get_recommendations(category: Optional[str] = None) -> dict:
    """Get optimization recommendations."""
    all_recs = insights.get_all_recommendations()
    if category:
        recs = all_recs.get(category, [])
        return {"recommendations": [asdict(r) for r in recs]}
    flat = []
    for cat, recs in all_recs.items():
        for r in recs:
            flat.append({"category": cat, **asdict(r)})
    return {"recommendations": flat}


@mcp.tool()
def analyze_root_cause(metric: str = "genai_errors_total", lookback: str = "1h") -> dict:
    """Perform root cause analysis."""
    result = insights.analyze_root_cause(metric, lookback)
    return asdict(result) if result else {"message": "No anomalies detected"}


@mcp.tool()
def check_sla_compliance() -> dict:
    """Check SLA compliance."""
    results = insights.check_sla_compliance()
    return {"slas": [asdict(r) for r in results], "breaches": [r.name for r in results if not r.is_compliant]}


@mcp.tool()
def generate_health_report() -> dict:
    """Generate health report."""
    return insights.generate_health_report()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
