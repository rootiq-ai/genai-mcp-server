"""HTTP Server for Grafana chatbot."""
import json
import logging
import os
import re
from dataclasses import asdict
from datetime import datetime
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from . import tools, analytics, insights
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


def fmt(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


def process_query(query: str) -> str:
    """Process natural language query."""
    q = query.lower().strip()
    app_match = re.search(r'(?:for|in|app)\s+["\']?(\w[\w-]*)["\']?', q)
    app_filter = app_match.group(1) if app_match else None

    # Natural language - broken/issues/problems/wrong
    if any(w in q for w in ["broken", "wrong", "issue", "problem", "failing", "down", "bad", "worried", "concern"]):
        r = insights.generate_health_report()
        s = r["summary"]
        status_emoji = "🟢" if r["health_status"] == "healthy" else "🟡" if r["health_status"] == "warning" else "🔴"
        
        if r["health_status"] == "healthy" and not r["anomalies"]:
            return f"""{status_emoji} Everything looks good!

No critical issues detected.

Quick Stats:
- Requests: {fmt(s['total_requests'])}
- Error Rate: {s['error_rate_percent']}%
- Latency: {s['avg_latency_ms']}ms
- Quality: {s['avg_quality']*100:.1f}%"""
        else:
            issues = "\n".join([f"- {i}" for i in r["issues"]]) if r["issues"] else "None"
            anomalies = "\n".join([f"- {a['message']}" for a in r["anomalies"][:3]]) if r["anomalies"] else "None"
            return f"""{status_emoji} Status: {r['health_status'].upper()}

Issues Found:
{issues}

Anomalies:
{anomalies}

Quick Stats:
- Requests: {fmt(s['total_requests'])}
- Error Rate: {s['error_rate_percent']}%
- Latency: {s['avg_latency_ms']}ms"""

    # Natural language - spending/expensive/budget/save money
    if any(w in q for w in ["spending", "expensive", "budget", "save money", "spend less", "reduce cost", "cheaper"]):
        recs = insights.get_cost_recommendations()
        c = tools.get_cost(app_filter)
        if not recs:
            return f"Current Spend: ${c['total_cost_usd']:.6f}\n\nNo cost optimization issues found. You're doing great!"
        
        lines = [f"- {r.title}\n  Action: {r.action}" for r in recs[:3]]
        return f"""Current Spend: ${c['total_cost_usd']:.6f}

Cost Optimization Tips:

{chr(10).join(lines)}"""

    # Natural language - slow/fast/speed/performance
    if any(w in q for w in ["slow", "fast", "speed", "taking long", "response time", "waiting"]):
        lat = tools.get_latency(app_filter)
        recs = insights.get_performance_recommendations()
        
        if not lat["latency_ms"]:
            return "No latency data available yet."
        
        lines = [f"- {l['model']}: {l.get('p05', 'N/A')}ms (p50), {l.get('p095', 'N/A')}ms (p95)" for l in lat["latency_ms"]]
        result = "Latency by Model:\n\n" + "\n".join(lines)
        
        if recs:
            result += "\n\nRecommendations:\n" + "\n".join([f"- {r.title}" for r in recs[:2]])
        return result

    # Natural language - how is/how are/how's doing
    if any(w in q for w in ["how is", "how are", "how's", "doing", "performing", "status", "going"]):
        r = insights.generate_health_report()
        s = r["summary"]
        status_emoji = "🟢" if r["health_status"] == "healthy" else "🟡" if r["health_status"] == "warning" else "🔴"
        return f"""{status_emoji} System Status: {r['health_status'].upper()}

Requests: {fmt(s['total_requests'])}
Cost: ${s['total_cost_usd']:.4f}
Error Rate: {s['error_rate_percent']}%
Avg Latency: {s['avg_latency_ms']}ms
Quality: {s['avg_quality']*100:.1f}%

Issues: {', '.join(r['issues']) or 'None'}
SLA Breaches: {', '.join(r['sla_breaches']) or 'None'}"""

    # Natural language - important/focus/priority/fix
    if any(w in q for w in ["important", "focus", "priority", "fix first", "should i", "need to"]):
        all_recs = insights.get_all_recommendations()
        critical = []
        for cat, recs in all_recs.items():
            for r in recs:
                if r.priority in ["critical", "high"]:
                    critical.append(f"- [{cat.upper()}] {r.title}\n  {r.action}")
        
        if not critical:
            return "No critical issues! Your system is running well."
        
        return "Top Priority Actions:\n\n" + "\n\n".join(critical[:5])

    # Natural language - full picture/everything/overview/tell me about
    if any(w in q for w in ["full picture", "everything", "overview", "tell me about", "complete", "all metrics"]):
        return process_query("show health report")

    # "How many" questions
    if "how many" in q:
        if "token" in q:
            t = tools.get_tokens(app_filter)
            return f"Total Tokens Used: {fmt(t['total'])}\n- Input: {fmt(t['total_input'])}\n- Output: {fmt(t['total_output'])}"
        if "request" in q:
            r = tools.get_requests(app_filter)
            return f"Total Requests: {fmt(r['total'])}"
        if "error" in q:
            e = tools.get_errors(app_filter)
            return f"Total Errors: {e['total_errors']} ({e['error_rate_percent']}% error rate)"
        if "security" in q or "attack" in q:
            s = tools.get_security(app_filter)
            return f"Security Events: {s['total_events']}"

    # "What is" / "What's" questions
    if any(w in q for w in ["what is", "what's", "whats"]):
        if "cost" in q:
            c = tools.get_cost(app_filter)
            return f"Total Cost: ${c['total_cost_usd']:.6f}"
        if "latency" in q:
            l = tools.get_latency(app_filter)
            if l["latency_ms"]:
                return f"Latency: {l['latency_ms'][0].get('p05', 'N/A')}ms (p50)"
            return "No latency data available"
        if "quality" in q:
            s = tools.get_summary(app_filter)["summary"]
            return f"Quality Score: {s['avg_quality']*100:.1f}%"
        if "error rate" in q:
            e = tools.get_errors(app_filter)
            return f"Error Rate: {e['error_rate_percent']}%"

    # Anomaly detection
    if any(w in q for w in ["anomal", "unusual", "spike", "outlier"]):
        metric = "genai_errors_total"
        if "latency" in q:
            metric = "genai_latency_seconds"
        elif "cost" in q:
            metric = "genai_cost_dollars_total"
        
        anomalies = analytics.detect_anomalies(metric, lookback="1h")
        if not anomalies:
            return f"No anomalies detected in {metric}"
        
        lines = [f"- {a.severity.upper()}: {a.message}" for a in anomalies[:5]]
        return "Anomaly Detection:\n\n" + "\n".join(lines)

    # Trends
    if any(w in q for w in ["trend", "trending", "direction"]):
        metric = "genai_requests_total"
        if "cost" in q:
            metric = "genai_cost_dollars_total"
        elif "error" in q:
            metric = "genai_errors_total"
        
        trends = analytics.analyze_trend(metric, lookback="6h")
        if not trends:
            return "Unable to analyze trends"
        
        t = trends[0]
        return f"Trend Analysis ({metric}):\n\nDirection: {t.direction}\nChange: {t.change_percent:+.1f}%\nR2: {t.r_squared:.2f}"

    # Forecast
    if any(w in q for w in ["forecast", "predict", "future"]):
        metric = "genai_cost_dollars_total"
        forecast = analytics.forecast_metric(metric, 24, "24h")
        if "error" in forecast:
            return forecast["error"]
        
        lines = [f"- {f['hours_ahead']}h: {f['predicted']:.4f}" for f in forecast["forecasts"][:6]]
        return f"Forecast ({metric}):\n\nCurrent: {forecast['current_value']:.4f}\nTrend: {forecast['trend']['direction']}\n\n" + "\n".join(lines)

    # Recommendations
    if any(w in q for w in ["recommend", "suggest", "optimize", "improve"]):
        category = None
        if "cost" in q:
            category = "cost"
        elif "quality" in q:
            category = "quality"
        elif "performance" in q:
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
        
        if not recs:
            return "No recommendations - everything looks good!"
        
        lines = [f"- [{r.priority.upper()}] {r.title}\n  {r.action}" for r in recs[:5]]
        return "Recommendations:\n\n" + "\n\n".join(lines)

    # Root cause
    if any(w in q for w in ["root cause", "why", "diagnose"]):
        result = insights.analyze_root_cause("genai_errors_total")
        if not result:
            return "No issues detected"
        
        causes = [f"- {c['cause']} ({c['confidence']*100:.0f}%)" for c in result.probable_causes[:3]]
        return f"Root Cause Analysis:\n\nIssue: {result.issue}\nSeverity: {result.severity}\n\nCauses:\n" + "\n".join(causes)

    # SLA
    if any(w in q for w in ["sla", "compliance", "target"]):
        slas = insights.check_sla_compliance()
        lines = []
        for s in slas:
            status = "PASS" if s.is_compliant else "FAIL"
            lines.append(f"- [{status}] {s.name}: {s.current:.2f} (target: {s.target})")
        return "SLA Compliance:\n\n" + "\n".join(lines)

    # Health report
    if any(w in q for w in ["health", "status", "report", "overview"]):
        r = insights.generate_health_report()
        s = r["summary"]
        return f"""Health Report: {r['health_status'].upper()}

Requests: {fmt(s['total_requests'])}
Cost: ${s['total_cost_usd']:.4f}
Error Rate: {s['error_rate_percent']}%
Latency: {s['avg_latency_ms']}ms
Quality: {s['avg_quality']*100:.1f}%

Issues: {', '.join(r['issues']) or 'None'}
SLA Breaches: {', '.join(r['sla_breaches']) or 'None'}"""

    # Summary
    if any(w in q for w in ["summary", "all", "everything"]):
        s = tools.get_summary(app_filter)["summary"]
        return f"""GenAI Summary:

Requests: {fmt(s['total_requests'])}
Cost: ${s['total_cost_usd']:.4f}
Tokens: {fmt(s['total_tokens'])}
Latency: {s['avg_latency_ms']}ms (p50), {s['p95_latency_ms']}ms (p95)
Quality: {s['avg_quality']*100:.1f}%
Error Rate: {s['error_rate_percent']}%
Security Events: {s['security_events']}"""

    # Cost
    if any(w in q for w in ["cost", "spend", "money"]):
        c = tools.get_cost(app_filter)
        breakdown = "\n".join([f"- {i.get('model', 'unknown')}: ${i['cost_usd']:.6f}" for i in c["breakdown"][:5]])
        return f"Cost: ${c['total_cost_usd']:.6f}\n\nBy Model:\n{breakdown or 'No data'}"

    # Quality
    if any(w in q for w in ["quality", "groundedness"]):
        data = tools.get_quality(app_filter)
        if not data["quality"]:
            return "No quality data"
        lines = [f"- {i['application']}: {i.get('overall', 0)*100:.1f}%" for i in data["quality"]]
        return "Quality:\n\n" + "\n".join(lines)

    # Security
    if any(w in q for w in ["security", "injection", "pii"]):
        sec = tools.get_security(app_filter)
        return f"Security Events: {sec['total_events']}"

    # Latency
    if any(w in q for w in ["latency", "slow", "speed"]):
        lat = tools.get_latency(app_filter)
        if not lat["latency_ms"]:
            return "No latency data"
        lines = [f"- {i['model']}: p50={i.get('p05', 'N/A')}ms p95={i.get('p095', 'N/A')}ms" for i in lat["latency_ms"]]
        return "Latency:\n\n" + "\n".join(lines)

    # Errors
    if any(w in q for w in ["error", "fail"]):
        err = tools.get_errors(app_filter)
        return f"Errors: {err['total_errors']} ({err['error_rate_percent']}%)"

    # Tokens
    if any(w in q for w in ["token", "tokens", "usage"]):
        t = tools.get_tokens(app_filter)
        return f"""Token Usage:

Total Tokens: {fmt(t['total'])}
- Input Tokens: {fmt(t['total_input'])}
- Output Tokens: {fmt(t['total_output'])}

Ratio: {t['total_output']/t['total_input']:.2f}x output/input""" if t['total_input'] > 0 else f"""Token Usage:

Total Tokens: {fmt(t['total'])}
- Input Tokens: {fmt(t['total_input'])}
- Output Tokens: {fmt(t['total_output'])}"""

    # Default
    return """I can help you with GenAI observability! Try asking:

📊 Status: "Is anything broken?" or "How is my system doing?"
💰 Costs: "How much am I spending?" or "How can I reduce costs?"
⚡ Performance: "Why are responses slow?"
🔍 Analysis: "Detect anomalies" or "Show trends"
📈 Forecast: "Forecast my costs"
💡 Advice: "What should I focus on?" or "Give me recommendations"
🏥 Health: "Show health report"

Just ask naturally!"""


@app.route("/")
def index():
    return jsonify({"name": "GenAI MCP Server (Advanced)", "version": "2.0.0", "status": "running"})


@app.route("/health")
def health():
    return jsonify({"status": "healthy" if prom.is_connected() else "degraded", "timestamp": datetime.now().isoformat()})


@app.route("/query", methods=["POST", "OPTIONS"])
def query():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_query = data.get("question") or data.get("query") or data.get("message") or ""
        logger.info(f"Query: {user_query}")
        if not user_query:
            return jsonify({"answer": "Ask about your GenAI metrics!", "response": "Ask about your GenAI metrics!"})
        response_text = process_query(user_query)
        return jsonify({"answer": response_text, "response": response_text})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e), "answer": f"Error: {e}"})


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    return query()


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


@app.route("/api/sla")
def api_sla():
    results = insights.check_sla_compliance()
    return jsonify({"slas": [asdict(r) for r in results]})


def main():
    banner = """
============================================================
  GenAI Observability MCP Server (ADVANCED)
============================================================
   Query:   http://{}:{}//query
   Health:  http://{}:{}/health
   
   Features: Anomaly Detection, Trends, Forecasting,
             Root Cause, Recommendations, SLA Monitoring
============================================================
""".format(HOST, PORT, HOST, PORT)
    print(banner)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
