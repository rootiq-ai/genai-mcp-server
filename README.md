# GenAI Observability MCP Server (Advanced)

Advanced MCP Server for GenAI/LLM observability with anomaly detection, trend analysis, forecasting, and intelligent insights.

## Features

### Basic Metrics
- Requests, Cost, Tokens, Latency, Quality, RAG, Security, Errors

### Advanced Analytics
- **Anomaly Detection** - Z-score based
- **Trend Analysis** - Linear regression
- **Forecasting** - Predictions with confidence intervals
- **Correlations** - Find related metrics

### Intelligent Insights
- **Cost Optimization** recommendations
- **Root Cause Analysis**
- **SLA Monitoring**
- **Health Reports**

## Installation

```bash
cd genai-mcp-server-advanced
uv sync
```

## Usage

### HTTP Server (Grafana)

```bash
uv run genai-mcp-http-advanced
```

Server starts at `http://localhost:3001`

### MCP Server (Claude Desktop)

```bash
uv run genai-mcp-advanced
```

## Grafana Configuration

Set MCP Server URL to: `http://localhost:3001`

## Example Queries

- "show health report"
- "detect anomalies"
- "show trends"
- "forecast cost"
- "recommend optimizations"
- "root cause analysis"
- "check SLA"
- "summary"

## Configuration

```bash
export PROMETHEUS_URL=http://localhost:9090
export MCP_PORT=3001
```

## License

MIT
