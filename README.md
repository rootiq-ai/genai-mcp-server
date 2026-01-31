# GenAI Observability MCP Server (Advanced + LLM + Charts)

Advanced MCP Server for GenAI/LLM observability with OpenAI-powered NLP, chart generation, anomaly detection, trend analysis, forecasting, and intelligent insights.

## Features

### 🤖 OpenAI LLM Integration
- Natural language understanding
- Intelligent response generation
- Context-aware answers

### 📊 Chart Generation (16 Types!)
- **Trends**: cost_trend, requests_trend, latency_trend, errors_trend, tokens_trend
- **Breakdowns**: cost_by_model, requests_by_model, latency_by_model, errors_by_type, quality_by_app
- **Analysis**: anomalies, trends, forecast
- **Overview**: dashboard, health_gauge, security_events

### 🔍 Advanced Analytics
- Anomaly Detection (Z-score based)
- Trend Analysis (Linear regression)
- Forecasting with confidence intervals
- Root Cause Analysis

### 💡 Intelligent Insights
- Cost Optimization recommendations
- SLA Monitoring
- Health Reports

## Installation

```bash
cd genai-mcp-server-advanced
uv sync
```

## Configuration

```bash
# Required for LLM features
export OPENAI_API_KEY="sk-your-key-here"

# Optional
export OPENAI_MODEL="gpt-3.5-turbo"  # or gpt-4o-mini
export PROMETHEUS_URL="http://localhost:9090"
export MCP_PORT="3001"
export USE_LLM="true"  # set to false to disable LLM
```

## Usage

```bash
uv run genai-mcp-http-advanced
```

## Chart Examples

### Natural Language Queries
- "Show me a dashboard chart"
- "Visualize cost trends"
- "Generate a latency comparison chart"
- "Chart errors over time"
- "Draw a forecast"

### Direct API Access
```bash
# Get chart as JSON (base64)
curl http://localhost:3001/api/chart/dashboard

# Get chart as PNG image
curl http://localhost:3001/api/chart/dashboard/image > dashboard.png

# List available charts
curl http://localhost:3001/api/charts
```

## Available Charts

| Chart Type | Description |
|------------|-------------|
| `dashboard` | Complete overview with all metrics |
| `cost_trend` | Cost over time |
| `cost_by_model` | Cost breakdown by model (pie) |
| `requests_trend` | Request rate over time |
| `requests_by_model` | Requests by model (bar) |
| `latency_trend` | P95 latency over time |
| `latency_by_model` | Latency comparison P50/P95/P99 |
| `tokens_trend` | Input/output tokens over time |
| `errors_trend` | Error rate over time |
| `errors_by_type` | Errors breakdown (pie) |
| `quality_by_app` | Quality scores by application |
| `security_events` | Security event counts |
| `health_gauge` | System health score gauge |
| `forecast` | Cost predictions with confidence |
| `anomalies` | Anomaly detection visualization |
| `trends` | Multi-metric trend analysis |

## License

MIT
