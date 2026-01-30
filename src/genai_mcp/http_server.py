#!/usr/bin/env python3
"""
Advanced HTTP Server for Grafana GenAI Chatbot.

Features:
- Natural language queries
- Advanced analytics (anomalies, trends, forecasting)
- Intelligent recommendations
- SLA monitoring
- Health reports

Usage:
    uv run genai-mcp-http-advanced
    
Grafana Config:
    MCP Server URL: http://localhost:3001
"""
import json
import logging
import os
import re
from dataclasses import asdict
from datetime import datetime
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

from . import tools
from . import analytics
from . import insights
from .prometheus import prom

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("MCP_PORT", "3001"))
HOST = os.getenv("MCP_HOST", "0.0.0.0")

app = Flask(__name__)
CORS(app)


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    return response


def fmt(n: float) -> str:
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


def process_query(query: str) -> str:
    """Process natural language query with advanced capabilities."""
    q = query.lower().strip()
    
    # Extract application filter
    app_match = re.search(r'(?:for|in|app)\s+["\']?(\w[\w-]*)["\']?', q)
    app = app_match.group(1) if app_match else None

    # =========================================================================
    # ADVANCED ANALYTICS
    # =========================================================================
    
    # Anomaly detection
    if any(w in q for w in ["anomal", "unusual", "abnormal", "spike", "outlier"]):
        metric = "genai_errors_total"
        if "latency" in q:
            metric = "genai_latency_seconds"
        elif "cost" in q:
            metric = "genai_cost_dollars_total"
        
        anomalies = analytics.detect_anomalies(metric, lookback="1h")
        if not anomalies:
            return f"✅ **No anomalies detected** in {metric} over the last hour."
        
        lines = []
        for a in anomalies[:5]:
            lines.append(f"- **{a.severity.upper()}**: {a.message}")
        
        return f"""🚨 **Anomaly Detection** ({metric})

Found {len(anomalies)} anomalies:

{chr(10).join(lines)}"""

    # Trend analysis
    if any(w in q for w in ["trend", "trending", "direction", "increasing", "decreasing"]):
        metric = "genai_requests_total"
        if "cost" in q:
            metric = "genai_cost_dollars_total"
        elif "error" in q:
            metric = "genai_errors_total"
        elif "latency" in q:
            metric = "genai_latency_seconds"
        
        trends = analytics.analyze_trend(metric, lookback="6h")
        if not trends:
            return "Unable to analyze trends. Insufficient data."
        
        t = trends[0]
        return f"""📈 **Trend Analysis** ({metric})

- **Direction**: {t.direction.upper()}
- **Change**: {t.change_percent:+.1f}%
- **R²**: {t.r_squared:.2f}
- **Forecast (1h)**: {t.forecast_1h:.4f}
- **Forecast (24h)**: {t.forecast_24h:.4f}"""

    # Forecasting
    if any(w in q for w in ["forecast", "predict", "projection", "future"]):
        metric = "genai_cost_dollars_total"
        hours = 24
        
        if "request" in q:
            metric = "genai_requests_total"
        
        forecast = analytics.forecast_metric(metric, hours, "24h")
        if "error" in forecast:
            return forecast["error"]
        
        lines = [f"- **{f['hours_ahead']}h**: {f['predicted']:.4f} (95% CI: {f['lower_95']:.4f} - {f['upper_95']:.4f})" 
                 for f in forecast["forecasts"][:6]]
        
        return f"""🔮 **Forecast** ({metric})

**Current**: {forecast['current_value']:.4f}
**Trend**: {forecast['trend']['direction']}

**Predictions:**
{chr(10).join(lines)}"""

    # Correlations
    if any(w in q for w in ["correlat", "related", "relationship"]):
        correlations = analytics.find_correlations(lookback="1h")
        if not correlations:
            return "No significant correlations found."
        
        lines = [f"- {c.metric_a} ↔ {c.metric_b}: **{c.correlation:.2f}** ({c.relationship})"
                 for c in correlations[:5]]
        
        return f"""🔗 **Metric Correlations**

{chr(10).join(lines)}"""

    # =========================================================================
    # INTELLIGENT INSIGHTS
    # =========================================================================
    
    # Recommendations
    if any(w in q for w in ["recommend", "suggestion", "advice", "optimize", "improve"]):
        category = None
        if "cost" in q:
            category = "cost"
        elif "quality" in q:
            category = "quality"
        elif "performance" in q or "latency" in q:
            category = "performance"
        elif "security" in q:
            category = "security"
        
        all_recs = insights.get_all_recommendations()
        
        if category:
            recs = all_recs.get(category, [])
        else:
            recs = []
            for r_list in all_recs.values():
                recs.extend(r_list)
            recs.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.priority, 4))
        
        if not recs:
            return "✅ **No recommendations** - Everything looks good!"
        
        lines = []
        for r in recs[:5]:
            emoji = "🔴" if r.priority == "critical" else "🟠" if r.priority == "high" else "🟡" if r.priority == "medium" else "🟢"
            lines.append(f"{emoji} **{r.title}**\n   {r.description}\n   💡 *{r.action}*")
        
        title = f"{category.title()} Recommendations" if category else "Top Recommendations"
        return f"""💡 **{title}**

{chr(10).join(lines)}"""

    # Root cause analysis
    if any(w in q for w in ["root cause", "why", "diagnose", "investigate"]):
        metric = "genai_errors_total"
        if "latency" in q:
            metric = "genai_latency_seconds"
        elif "cost" in q:
            metric = "genai_cost_dollars_total"
        
        result = insights.analyze_root_cause(metric)
        if not result:
            return f"✅ No issues detected in {metric} requiring root cause analysis."
        
        causes = [f"- **{c['cause']}** (confidence: {c['confidence']*100:.0f}%)\n  Evidence: {c['evidence']}" 
                  for c in result.probable_causes[:3]]
        recs = [f"- {r}" for r in result.recommendations[:3]]
        
        return f"""🔍 **Root Cause Analysis**

**Issue**: {result.issue}
**Severity**: {result.severity.upper()}

**Probable Causes:**
{chr(10).join(causes) or 'None identified'}

**Recommendations:**
{chr(10).join(recs) or 'Monitor the situation'}"""

    # SLA compliance
    if any(w in q for w in ["sla", "compliance", "target", "breach"]):
        slas = insights.check_sla_compliance()
        
        lines = []
        for s in slas:
            status = "✅" if s.is_compliant else "❌"
            trend_emoji = "📈" if s.trend == "increasing" else "📉" if s.trend == "decreasing" else "➡️"
            breach_info = f" ⚠️ {s.time_to_breach}" if s.time_to_breach else ""
            lines.append(f"{status} **{s.name}**: {s.current:.2f} (target: {s.target}){breach_info} {trend_emoji}")
        
        compliant = sum(1 for s in slas if s.is_compliant)
        return f"""📊 **SLA Compliance** ({compliant}/{len(slas)} passing)

{chr(10).join(lines)}"""

    # Health report
    if any(w in q for w in ["health", "status", "overview", "report"]):
        report = insights.generate_health_report()
        
        status_emoji = "🟢" if report["health_status"] == "healthy" else "🟡" if report["health_status"] == "warning" else "🔴"
        s = report["summary"]
        
        issues = ", ".join(report["issues"]) if report["issues"] else "None"
        breaches = ", ".join(report["sla_breaches"]) if report["sla_breaches"] else "None"
        
        anomaly_lines = [f"- {a['message']}" for a in report["anomalies"][:3]]
        rec_lines = [f"- [{r['category']}] {r['title']}" for r in report["critical_recommendations"][:3]]
        
        return f"""{status_emoji} **System Health Report**

**Status**: {report["health_status"].upper()}

**Summary:**
| Metric | Value |
|--------|-------|
| Requests | {fmt(s['total_requests'])} |
| Cost | ${s['total_cost_usd']:.4f} |
| Error Rate | {s['error_rate_percent']}% |
| Latency (p50) | {s['avg_latency_ms']}ms |
| Quality | {s['avg_quality']*100:.1f}% |

**Issues**: {issues}
**SLA Breaches**: {breaches}

**Anomalies:**
{chr(10).join(anomaly_lines) or '- None detected'}

**Top Actions:**
{chr(10).join(rec_lines) or '- No critical actions needed'}"""

    # =========================================================================
    # BASIC METRICS (fallback to standard queries)
    # =========================================================================
    
    if any(w in q for w in ["summary", "all", "everything", "dashboard"]):
        s = tools.get_summary(app)["summary"]
        return f"""📊 **GenAI Metrics Summary**

| Metric | Value |
|--------|-------|
| Requests | {fmt(s['total_requests'])} |
| Cost | ${s['total_cost_usd']:.4f} |
| Tokens | {fmt(s['total_tokens'])} |
| Latency (p50) | {s['avg_latency_ms']}ms |
| Latency (p95) | {s['p95_latency_ms']}ms |
| Quality | {s['avg_quality']*100:.1f}% |
| Error Rate | {s['error_rate_percent']}% |
| Security Events | {s['security_events']} |"""

    if any(w in q for w in ["cost", "spend", "money", "dollar"]):
        c = tools.get_cost(app)
        breakdown = "\n".join([f"- {i.get('model', 'unknown')}: ${i['cost_usd']:.6f}" for i in c["breakdown"][:5]])
        return f"💰 **Cost**: ${c['total_cost_usd']:.6f}\n\n**By Model:**\n{breakdown or 'No data'}"

    if any(w in q for w in ["quality", "groundedness", "relevance"]):
        data = tools.get_quality(app)
        if not data["quality"]: return "No quality data."
        lines = [f"- **{i['application']}**: {i.get('overall', 0)*100:.1f}%" for i in data["quality"]]
        return "✅ **Quality**\n\n" + "\n".join(lines)

    if any(w in q for w in ["security", "injection", "pii"]):
        sec = tools.get_security(app)
        return f"🔒 **Security**: {sec['total_events']} events"

    if any(w in q for w in ["latency", "slow", "speed"]):
        lat = tools.get_latency(app)
        if not lat["latency_ms"]: return "No latency data."
        lines = [f"- {i['model']}: p50={i.get('p05', 'N/A')}ms p95={i.get('p095', 'N/A')}ms" for i in lat["latency_ms"]]
        return "⚡ **Latency**\n\n" + "\n".join(lines)

    if any(w in q for w in ["error", "fail"]):
        err = tools.get_errors(app)
        return f"❌ **Errors**: {err['total_errors']} total"

    # Default - show health report
    report = insights.generate_health_report()
    s = report["summary"]
    return f"""Ask me about: anomalies, trends, forecasts, recommendations, root cause, SLA, health

**Quick Status** ({report["health_status"]}):
- Requests: {fmt(s['total_requests'])}
- Cost: ${s['total_cost_usd']:.4f}
- Quality: {s['avg_quality']*100:.1f}%
- Errors: {s['error_rate_percent']}%

Try: "detect anomalies", "show trends", "recommend optimizations", "check SLA""""


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/")
def index():
    return jsonify({
        "name": "GenAI MCP Server (Advanced)",
        "version": "2.0.0",
        "features": ["anomaly_detection", "trend_analysis", "forecasting", "recommendations", "root_cause", "sla_monitoring"],
    })


@app.route("/health")
def health():
    report = insights.generate_health_report()
    return jsonify({
        "status": report["health_status"],
        "prometheus_connected": prom.is_connected(),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/query", methods=["POST", "OPTIONS"])
def query():
    """Handle natural language queries."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_query = data.get("question") or data.get("query") or data.get("message") or ""
        logger.info(f"Query: {user_query}")
        
        if not user_query:
            return jsonify({"answer": "Ask about anomalies, trends, forecasts, recommendations, SLA, or health!"})
        
        response_text = process_query(user_query)
        return jsonify({"answer": response_text, "response": response_text})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e), "answer": f"Error: {e}"})


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    return query()


# REST API endpoints
@app.route("/api/summary")
def api_summary():
    return jsonify(tools.get_summary(request.args.get("application")))


@app.route("/api/health-report")
def api_health():
    return jsonify(insights.generate_health_report())


@app.route("/api/recommendations")
def api_recommendations():
    all_recs = insights.get_all_recommendations()
    return jsonify({cat: [asdict(r) for r in recs] for cat, recs in all_recs.items()})


@app.route("/api/anomalies/<metric>")
def api_anomalies(metric):
    results = analytics.detect_anomalies(metric, lookback=request.args.get("lookback", "1h"))
    return jsonify({"anomalies": [asdict(r) for r in results]})


@app.route("/api/trends/<metric>")
def api_trends(metric):
    results = analytics.analyze_trend(metric, lookback=request.args.get("lookback", "6h"))
    return jsonify({"trends": [asdict(r) for r in results]})


@app.route("/api/forecast/<metric>")
def api_forecast(metric):
    return jsonify(analytics.forecast_metric(metric, int(request.args.get("hours", 24))))


@app.route("/api/sla")
def api_sla():
    results = insights.check_sla_compliance()
    return jsonify({"slas": [asdict(r) for r in results]})


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"""
{'='*60}
🚀 GenAI Observability MCP Server (ADVANCED)
{'='*60}
   Query:   http://{HOST}:{PORT}/query
   Health:  http://{HOST}:{PORT}/health
   
   ADVANCED FEATURES:
   • Anomaly Detection
   • Trend Analysis  
   • Forecasting
   • Root Cause Analysis
   • Recommendations
   • SLA Monitoring
{'='*60}
📋 Grafana: Set MCP URL to http://localhost:{PORT}
{'='*60}
""")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
