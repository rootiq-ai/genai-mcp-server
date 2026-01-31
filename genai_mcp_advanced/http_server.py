"""HTTP Server for Grafana chatbot - With OpenAI LLM + Chart Generation."""
import json
import logging
import os
import re
from dataclasses import asdict
from datetime import datetime
from flask import Flask, jsonify, make_response, request, Response
from flask_cors import CORS
from . import tools, analytics, insights, charts
from .prometheus import prom

# OpenAI Integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Run: pip install openai")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("MCP_PORT", "3001"))
HOST = os.getenv("MCP_HOST", "0.0.0.0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
openai_client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI client initialized")
else:
    logger.warning("⚠️ OpenAI not configured. Set OPENAI_API_KEY environment variable.")


# =============================================================================
# CORS HANDLERS
# =============================================================================

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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fmt(n):
    """Format numbers nicely."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


# =============================================================================
# CHART DETECTION & GENERATION
# =============================================================================

CHART_KEYWORDS = {
    'chart': True, 'graph': True, 'plot': True, 'visualize': True, 
    'visualization': True, 'show me a chart': True, 'draw': True,
    'diagram': True, 'visual': True, 'picture': True, 'image': True
}

CHART_TYPE_MAPPING = {
    # Cost charts
    ('cost', 'trend'): 'cost_trend',
    ('cost', 'over time'): 'cost_trend',
    ('cost', 'history'): 'cost_trend',
    ('cost', 'model'): 'cost_by_model',
    ('cost', 'breakdown'): 'cost_by_model',
    ('spending', 'trend'): 'cost_trend',
    ('spending', 'model'): 'cost_by_model',
    
    # Request charts
    ('request', 'trend'): 'requests_trend',
    ('request', 'over time'): 'requests_trend',
    ('request', 'model'): 'requests_by_model',
    ('traffic', 'trend'): 'requests_trend',
    
    # Latency charts
    ('latency', 'trend'): 'latency_trend',
    ('latency', 'over time'): 'latency_trend',
    ('latency', 'model'): 'latency_by_model',
    ('latency', 'comparison'): 'latency_by_model',
    ('performance', 'trend'): 'latency_trend',
    ('speed', 'trend'): 'latency_trend',
    
    # Token charts
    ('token', 'trend'): 'tokens_trend',
    ('token', 'over time'): 'tokens_trend',
    ('usage', 'trend'): 'tokens_trend',
    
    # Error charts
    ('error', 'trend'): 'errors_trend',
    ('error', 'over time'): 'errors_trend',
    ('error', 'type'): 'errors_by_type',
    ('error', 'breakdown'): 'errors_by_type',
    ('failure', 'trend'): 'errors_trend',
    
    # Quality charts
    ('quality', 'app'): 'quality_by_app',
    ('quality', 'application'): 'quality_by_app',
    ('quality', 'comparison'): 'quality_by_app',
    
    # Security charts
    ('security', ''): 'security_events',
    ('security', 'event'): 'security_events',
    ('attack', ''): 'security_events',
    
    # Special charts
    ('health', ''): 'health_gauge',
    ('health', 'score'): 'health_gauge',
    ('health', 'gauge'): 'health_gauge',
    ('forecast', ''): 'forecast',
    ('predict', ''): 'forecast',
    ('anomaly', ''): 'anomalies',
    ('anomalies', ''): 'anomalies',
    ('trend', 'all'): 'trends',
    ('trends', ''): 'trends',
    ('dashboard', ''): 'dashboard',
    ('overview', ''): 'dashboard',
    ('all', 'chart'): 'dashboard',
}


def detect_chart_request(query: str) -> tuple[bool, str]:
    """Detect if query is asking for a chart and determine chart type."""
    q = query.lower()
    
    # Check if it's a chart request
    is_chart_request = any(keyword in q for keyword in CHART_KEYWORDS)
    
    if not is_chart_request:
        return False, None
    
    # Determine chart type
    for (key1, key2), chart_type in CHART_TYPE_MAPPING.items():
        if key1 in q and (not key2 or key2 in q):
            return True, chart_type
    
    # Default to dashboard if chart requested but type unclear
    if is_chart_request:
        if 'cost' in q:
            return True, 'cost_trend'
        if 'latency' in q or 'performance' in q or 'slow' in q:
            return True, 'latency_trend'
        if 'error' in q:
            return True, 'errors_trend'
        if 'request' in q or 'traffic' in q:
            return True, 'requests_trend'
        if 'token' in q:
            return True, 'tokens_trend'
        if 'quality' in q:
            return True, 'quality_by_app'
        if 'security' in q:
            return True, 'security_events'
        if 'health' in q:
            return True, 'health_gauge'
        if 'forecast' in q or 'predict' in q:
            return True, 'forecast'
        if 'anomal' in q:
            return True, 'anomalies'
        if 'trend' in q:
            return True, 'trends'
        
        # Default to dashboard
        return True, 'dashboard'
    
    return False, None


def generate_chart_response(chart_type: str) -> dict:
    """Generate chart and return response with base64 image."""
    try:
        chart_base64 = charts.generate_chart(chart_type)
        
        if chart_base64:
            # Create HTML with embedded image
            html_response = f"""📊 Here's your {chart_type.replace('_', ' ')} chart:

<img src="data:image/png;base64,{chart_base64}" alt="{chart_type}" style="max-width:100%;"/>

*Chart generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
            
            return {
                "answer": html_response,
                "response": html_response,
                "chart": chart_base64,
                "chart_type": chart_type,
                "has_chart": True
            }
        else:
            return {
                "answer": f"Unable to generate {chart_type} chart. No data available.",
                "response": f"Unable to generate {chart_type} chart. No data available.",
                "has_chart": False
            }
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return {
            "answer": f"Error generating chart: {str(e)}",
            "response": f"Error generating chart: {str(e)}",
            "has_chart": False
        }


# =============================================================================
# LLM FUNCTIONS
# =============================================================================

def get_all_metrics_context() -> str:
    """Get all metrics as context for LLM."""
    try:
        summary = tools.get_summary()["summary"]
        health = insights.generate_health_report()
        
        context = f"""
CURRENT GENAI METRICS:
- Total Requests: {summary.get('total_requests', 0)}
- Total Cost: ${summary.get('total_cost_usd', 0):.6f}
- Total Input Tokens: {summary.get('total_input_tokens', 0)}
- Total Output Tokens: {summary.get('total_output_tokens', 0)}
- Total Tokens: {summary.get('total_tokens', 0)}
- Average Latency (P50): {summary.get('avg_latency_ms', 0)}ms
- P95 Latency: {summary.get('p95_latency_ms', 0)}ms
- Average Quality Score: {summary.get('avg_quality', 0)*100:.1f}%
- Groundedness: {summary.get('avg_groundedness', 0)*100:.1f}%
- Relevance: {summary.get('avg_relevance', 0)*100:.1f}%
- Coherence: {summary.get('avg_coherence', 0)*100:.1f}%
- Fluency: {summary.get('avg_fluency', 0)*100:.1f}%
- Error Rate: {summary.get('error_rate_percent', 0)}%
- Total Errors: {summary.get('total_errors', 0)}
- Security Events: {summary.get('security_events', 0)}

SYSTEM HEALTH:
- Status: {health.get('health_status', 'unknown').upper()}
- Issues: {', '.join(health.get('issues', [])) or 'None'}
- SLA Breaches: {', '.join(health.get('sla_breaches', [])) or 'None'}

AVAILABLE CHART TYPES:
{', '.join(charts.list_available_charts())}
"""
        return context
    except Exception as e:
        logger.error(f"Error getting metrics context: {e}")
        return "Unable to fetch current metrics."


SYSTEM_PROMPT = """You are an AI assistant for GenAI/LLM Observability. You help users understand their LLM application metrics.

You have access to real-time metrics from Prometheus including:
- Request counts and costs
- Token usage (input/output)
- Latency (P50, P95, P99)
- Quality scores (groundedness, relevance, coherence, fluency)
- Security events (prompt injection, PII, jailbreak attempts)
- Error rates and types

You can also generate CHARTS. When user asks for a chart, visualization, or graph, respond with:
"I'll generate a [chart_type] chart for you."

Available chart types: cost_trend, cost_by_model, requests_trend, requests_by_model, latency_trend, latency_by_model, tokens_trend, errors_trend, errors_by_type, quality_by_app, security_events, health_gauge, forecast, anomalies, trends, dashboard

RESPONSE GUIDELINES:
1. Be concise but informative
2. Use emojis sparingly for visual clarity
3. Format numbers nicely (1000 → 1K)
4. If user asks for a chart, mention that you're generating it
5. Always base answers on provided metrics data"""


def llm_generate_response(query: str, context: str) -> str:
    """Use LLM to generate a natural response."""
    if not openai_client:
        return None
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"""Based on the following metrics data, answer the user's question.

{context}

User Question: {query}

Provide a helpful, accurate response based ONLY on the data above."""}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM response generation error: {e}")
        return None


def process_with_llm(query: str) -> str:
    """Process query using LLM."""
    if not openai_client or not USE_LLM:
        return None
    
    context = get_all_metrics_context()
    return llm_generate_response(query, context)


# =============================================================================
# RULE-BASED FALLBACK
# =============================================================================

def process_rule_based(query: str) -> str:
    """Process query using rule-based keyword matching."""
    q = query.lower().strip()
    app_match = re.search(r'(?:for|in|app)\s+["\']?(\w[\w-]*)["\']?', q)
    app_filter = app_match.group(1) if app_match else None

    # Health/Status
    if any(w in q for w in ["health", "status", "ok", "okay", "working", "broken", "issue", "problem", "wrong", "down", "failing"]):
        r = insights.generate_health_report()
        s = r["summary"]
        emoji = "🟢" if r["health_status"] == "healthy" else "🟡" if r["health_status"] == "warning" else "🔴"
        return f"""{emoji} System Status: {r['health_status'].upper()}

Requests: {fmt(s['total_requests'])}
Cost: ${s['total_cost_usd']:.4f}
Error Rate: {s['error_rate_percent']}%
Latency: {s['avg_latency_ms']}ms
Quality: {s['avg_quality']*100:.1f}%

Issues: {', '.join(r['issues']) or 'None'}
SLA Breaches: {', '.join(r['sla_breaches']) or 'None'}

💡 Tip: Ask for a "dashboard chart" to visualize all metrics!"""

    # Cost
    if any(w in q for w in ["cost", "spend", "spending", "money", "dollar", "price", "budget", "expensive"]):
        c = tools.get_cost(app_filter)
        breakdown = "\n".join([f"- {i.get('model', 'unknown')}: ${i['cost_usd']:.6f}" for i in c["breakdown"][:5]])
        return f"""💰 Total Cost: ${c['total_cost_usd']:.6f}

By Model:
{breakdown or 'No data'}

💡 Tip: Ask for a "cost trend chart" or "cost by model chart"!"""

    # Tokens
    if any(w in q for w in ["token", "tokens", "usage"]):
        t = tools.get_tokens(app_filter)
        ratio = f"\nRatio: {t['total_output']/t['total_input']:.2f}x output/input" if t['total_input'] > 0 else ""
        return f"""📊 Token Usage:

Total: {fmt(t['total'])}
- Input: {fmt(t['total_input'])}
- Output: {fmt(t['total_output'])}{ratio}

💡 Tip: Ask for a "token trend chart"!"""

    # Latency
    if any(w in q for w in ["latency", "slow", "fast", "speed", "performance", "response time"]):
        lat = tools.get_latency(app_filter)
        if not lat["latency_ms"]:
            return "No latency data available"
        lines = [f"- {l['model']}: p50={l.get('p05', 'N/A')}ms, p95={l.get('p095', 'N/A')}ms" for l in lat["latency_ms"]]
        return "⚡ Latency by Model:\n\n" + "\n".join(lines) + "\n\n💡 Tip: Ask for a 'latency chart'!"

    # Quality
    if any(w in q for w in ["quality", "groundedness", "relevance", "coherence", "fluency", "accurate"]):
        data = tools.get_quality(app_filter)
        if not data["quality"]:
            return "No quality data available"
        lines = [f"- {i['application']}: {i.get('overall', 0)*100:.1f}%" for i in data["quality"]]
        return "✅ Quality Scores:\n\n" + "\n".join(lines) + "\n\n💡 Tip: Ask for a 'quality chart'!"

    # Security
    if any(w in q for w in ["security", "injection", "pii", "jailbreak", "attack"]):
        sec = tools.get_security(app_filter)
        triggers = "\n".join([f"- {t['type']}: {t['count']}" for t in sec["guardrail_triggers"]]) if sec["guardrail_triggers"] else "None"
        return f"🔒 Security Events: {sec['total_events']}\n\nGuardrail Triggers:\n{triggers}\n\n💡 Tip: Ask for a 'security chart'!"

    # Errors
    if any(w in q for w in ["error", "errors", "fail", "failure", "exception"]):
        err = tools.get_errors(app_filter)
        error_list = "\n".join([f"- {e['error_type']}: {e['count']}" for e in err["errors"]]) if err["errors"] else "None"
        return f"❌ Errors: {err['total_errors']} ({err['error_rate_percent']}%)\n\nBy Type:\n{error_list}\n\n💡 Tip: Ask for an 'error chart'!"

    # Anomalies
    if any(w in q for w in ["anomaly", "anomalies", "unusual", "spike", "outlier", "abnormal"]):
        all_anomalies = []
        for metric in ["genai_errors_total", "genai_latency_seconds", "genai_cost_dollars_total"]:
            results = analytics.detect_anomalies(metric, lookback="1h")
            all_anomalies.extend(results)
        
        if not all_anomalies:
            return "✅ No anomalies detected in the last hour.\n\n💡 Tip: Ask for an 'anomaly chart'!"
        
        lines = [f"- {a.severity.upper()}: {a.message}" for a in all_anomalies[:10]]
        return f"🚨 Anomalies Detected:\n\n" + "\n".join(lines) + "\n\n💡 Tip: Ask for an 'anomaly chart'!"

    # Trends
    if any(w in q for w in ["trend", "trending", "direction", "increasing", "decreasing"]):
        all_trends = []
        for metric in ["genai_requests_total", "genai_cost_dollars_total", "genai_errors_total"]:
            results = analytics.analyze_trend(metric, lookback="6h")
            all_trends.extend(results)
        
        if not all_trends:
            return "Unable to analyze trends (insufficient data)"
        
        lines = [f"- {t.metric_name}: {t.direction} ({t.change_percent:+.1f}%)" for t in all_trends]
        return "📈 Trend Analysis:\n\n" + "\n".join(lines) + "\n\n💡 Tip: Ask for a 'trends chart'!"

    # Forecast
    if any(w in q for w in ["forecast", "predict", "prediction", "future", "will be", "next"]):
        forecast = analytics.forecast_metric("genai_cost_dollars_total", 24, "24h")
        if "error" in forecast:
            return f"Unable to forecast: {forecast['error']}"
        
        lines = [f"- {f['hours_ahead']}h: ${f['predicted']:.6f}" for f in forecast["forecasts"][:6]]
        return f"🔮 Cost Forecast:\n\nCurrent: ${forecast['current_value']:.6f}\nTrend: {forecast['trend']['direction']}\n\n" + "\n".join(lines) + "\n\n💡 Tip: Ask for a 'forecast chart'!"

    # Recommendations
    if any(w in q for w in ["recommend", "suggestion", "optimize", "improve", "tip", "advice"]):
        all_recs = insights.get_all_recommendations()
        flat = []
        for cat, recs in all_recs.items():
            for r in recs:
                flat.append((cat, r))
        
        if not flat:
            return "✅ No recommendations - everything looks good!"
        
        lines = [f"- [{cat.upper()}] {r.title}\n  {r.action}" for cat, r in flat[:5]]
        return "💡 Recommendations:\n\n" + "\n\n".join(lines)

    # SLA
    if any(w in q for w in ["sla", "compliance", "target", "breach"]):
        slas = insights.check_sla_compliance()
        lines = []
        for s in slas:
            status = "✅" if s.is_compliant else "❌"
            lines.append(f"{status} {s.name}: {s.current:.2f} (target: {s.target})")
        return "📊 SLA Compliance:\n\n" + "\n".join(lines)

    # Summary
    if any(w in q for w in ["summary", "overview", "all", "everything"]):
        s = tools.get_summary(app_filter)["summary"]
        return f"""📊 GenAI Summary:

Requests: {fmt(s['total_requests'])}
Cost: ${s['total_cost_usd']:.6f}
Tokens: {fmt(s['total_tokens'])}
Latency: {s['avg_latency_ms']}ms (p50), {s['p95_latency_ms']}ms (p95)
Quality: {s['avg_quality']*100:.1f}%
Error Rate: {s['error_rate_percent']}%
Security Events: {s['security_events']}

💡 Tip: Ask for a "dashboard chart" to visualize everything!"""

    # Requests
    if any(w in q for w in ["request", "requests", "calls", "invocations"]):
        r = tools.get_requests(app_filter)
        lines = [f"- {req['model']}: {fmt(req['count'])}" for req in r["requests"]]
        return f"📈 Total Requests: {fmt(r['total'])}\n\nBy Model:\n" + "\n".join(lines) + "\n\n💡 Tip: Ask for a 'requests chart'!"

    # Models
    if any(w in q for w in ["model", "models", "which model", "list model"]):
        m = tools.get_models()
        if not m["models"]:
            return "No models found"
        return "🤖 Models in use:\n\n" + "\n".join([f"- {model}" for model in m["models"]])

    # Applications
    if any(w in q for w in ["application", "applications", "app", "apps", "which app"]):
        a = tools.get_applications()
        if not a["applications"]:
            return "No applications found"
        return "📱 Applications:\n\n" + "\n".join([f"- {app}" for app in a["applications"]])

    # Available charts
    if "available chart" in q or "what chart" in q or "list chart" in q:
        chart_list = charts.list_available_charts()
        return f"""📊 Available Charts:

{chr(10).join([f'- {c}' for c in chart_list])}

Example queries:
- "Show me a cost trend chart"
- "Generate a dashboard chart"
- "Visualize latency by model"
- "Chart errors over time"
- "Draw a quality comparison"""

    # Greetings
    if any(w in q for w in ["hello", "hi", "hey", "help", "what can you do"]):
        return """👋 Hello! I'm your GenAI Observability Assistant.

I can help you with:
📊 **Metrics**: requests, cost, tokens, latency, quality, errors
🔍 **Analysis**: anomalies, trends, forecasts, root cause
💡 **Insights**: recommendations, SLA compliance, health reports
📈 **Charts**: visualize any metric!

**Chart Commands:**
- "Show me a dashboard chart"
- "Visualize cost trends"
- "Chart latency by model"
- "Generate error breakdown chart"
- "Draw a forecast chart"

How can I help you today?"""

    # Default
    return """I'm not sure I understood that. Try asking about:

📊 **Metrics**: "summary", "cost", "latency", "quality", "errors"
📈 **Charts**: "show me a dashboard chart" or "visualize cost trends"
🔍 **Analysis**: "detect anomalies", "show trends", "forecast"
💡 **Advice**: "recommendations", "SLA compliance"

Type "available charts" to see all chart options!"""


# =============================================================================
# MAIN QUERY PROCESSOR
# =============================================================================

def process_query(query: str) -> dict:
    """Process query - detect charts, try LLM, fallback to rule-based."""
    
    # First, check if it's a chart request
    is_chart, chart_type = detect_chart_request(query)
    
    if is_chart and chart_type:
        logger.info(f"Chart request detected: {chart_type}")
        return generate_chart_response(chart_type)
    
    # Try LLM if available
    if openai_client and USE_LLM:
        try:
            response = process_with_llm(query)
            if response:
                return {"answer": response, "response": response, "has_chart": False}
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
    
    # Fallback to rule-based
    response = process_rule_based(query)
    return {"answer": response, "response": response, "has_chart": False}


# =============================================================================
# HTTP ROUTES
# =============================================================================

@app.route("/")
def index():
    return jsonify({
        "name": "GenAI MCP Server (Advanced + LLM + Charts)",
        "version": "2.2.0",
        "status": "running",
        "llm_enabled": bool(openai_client and USE_LLM),
        "llm_model": OPENAI_MODEL if openai_client else None,
        "charts_available": charts.list_available_charts()
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if prom.is_connected() else "degraded",
        "prometheus_connected": prom.is_connected(),
        "llm_available": bool(openai_client),
        "timestamp": datetime.now().isoformat()
    })


@app.route("/query", methods=["POST", "OPTIONS"])
def query():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_query = data.get("question") or data.get("query") or data.get("message") or ""
        logger.info(f"Query: {user_query}")
        
        if not user_query:
            return jsonify({
                "answer": "Ask me anything about your GenAI metrics! Try 'show me a dashboard chart'",
                "response": "Ask me anything about your GenAI metrics! Try 'show me a dashboard chart'",
                "has_chart": False
            })
        
        result = process_query(user_query)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e), "answer": f"Error: {e}", "has_chart": False})


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    return query()


# =============================================================================
# CHART API ENDPOINTS
# =============================================================================

@app.route("/api/chart/<chart_type>")
def api_chart(chart_type):
    """Generate and return a specific chart as PNG."""
    try:
        chart_base64 = charts.generate_chart(chart_type)
        if chart_base64:
            return jsonify({
                "chart": chart_base64,
                "chart_type": chart_type,
                "timestamp": datetime.now().isoformat()
            })
        return jsonify({"error": f"Unknown chart type: {chart_type}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/<chart_type>/image")
def api_chart_image(chart_type):
    """Return chart as actual PNG image."""
    try:
        import base64
        chart_base64 = charts.generate_chart(chart_type)
        if chart_base64:
            image_data = base64.b64decode(chart_base64)
            return Response(image_data, mimetype='image/png')
        return "Chart not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route("/api/charts")
def api_charts_list():
    """List all available charts."""
    return jsonify({
        "available_charts": charts.list_available_charts(),
        "examples": {
            "cost_trend": "/api/chart/cost_trend",
            "dashboard": "/api/chart/dashboard",
            "latency_by_model": "/api/chart/latency_by_model",
        }
    })


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.route("/api/summary")
def api_summary():
    return jsonify(tools.get_summary(request.args.get("application")))


@app.route("/api/health-report")
def api_health_report():
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


@app.route("/api/cost")
def api_cost():
    return jsonify(tools.get_cost(request.args.get("application")))


@app.route("/api/tokens")
def api_tokens():
    return jsonify(tools.get_tokens(request.args.get("application")))


@app.route("/api/latency")
def api_latency():
    return jsonify(tools.get_latency(request.args.get("application")))


@app.route("/api/quality")
def api_quality():
    return jsonify(tools.get_quality(request.args.get("application")))


@app.route("/api/security")
def api_security():
    return jsonify(tools.get_security(request.args.get("application")))


@app.route("/api/errors")
def api_errors():
    return jsonify(tools.get_errors(request.args.get("application")))


# =============================================================================
# MAIN
# =============================================================================

def main():
    llm_status = "✅ Enabled" if (openai_client and USE_LLM) else "❌ Disabled"
    llm_model = OPENAI_MODEL if openai_client else "N/A"
    
    banner = f"""
╔══════════════════════════════════════════════════════════════════╗
║     GenAI Observability MCP Server (Advanced + LLM + Charts)     ║
╠══════════════════════════════════════════════════════════════════╣
║  Query:   http://{HOST}:{PORT}/query                                
║  Health:  http://{HOST}:{PORT}/health                               
║  Charts:  http://{HOST}:{PORT}/api/charts                           
║                                                                  
║  LLM:     {llm_status}                                           
║  Model:   {llm_model}                                            
║                                                                  
║  Features:                                                       
║  • Natural Language Understanding (LLM-powered)                  
║  • 📊 Chart Generation (16 chart types!)                         
║  • Anomaly Detection                                             
║  • Trend Analysis & Forecasting                                  
║  • Root Cause Analysis                                           
║  • Recommendations                                               
╚══════════════════════════════════════════════════════════════════╝

📊 Available Charts: {', '.join(charts.list_available_charts())}

Example queries:
  • "Show me a dashboard chart"
  • "Visualize cost trends"
  • "Generate latency comparison chart"

To enable LLM: export OPENAI_API_KEY=your-key-here
"""
    print(banner)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
