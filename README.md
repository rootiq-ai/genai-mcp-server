# 🤖 GenAI Observability MCP Server - Advanced Edition

Advanced MCP Server for GenAI/LLM observability with **anomaly detection**, **trend analysis**, **forecasting**, **root cause analysis**, **intelligent recommendations**, and **SLA monitoring**.

Built with [FastMCP](https://github.com/jlowin/fastmcp) and integrates with Grafana chatbot.

## ✨ Features

### Basic Metrics
- Requests, Cost, Tokens, Latency, Quality, RAG, Security, Errors

### 🔬 Advanced Analytics
| Feature | Description |
|---------|-------------|
| **Anomaly Detection** | Z-score based detection with severity levels |
| **Trend Analysis** | Linear regression with R² and direction |
| **Forecasting** | Future predictions with confidence intervals |
| **Correlations** | Find related metrics with Pearson correlation |
| **Period Comparison** | Compare time periods with t-tests |
| **Percentiles** | Calculate p50/p90/p95/p99 |

### 💡 Intelligent Insights
| Feature | Description |
|---------|-------------|
| **Cost Optimization** | Identify expensive models, wasted spend |
| **Quality Recommendations** | Fix low quality scores |
| **Performance Recommendations** | Address latency issues |
| **Security Recommendations** | Handle threats |
| **Root Cause Analysis** | Diagnose why metrics are abnormal |
| **SLA Monitoring** | Track compliance, predict breaches |
| **Health Reports** | Comprehensive system status |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/genai-mcp-server-advanced.git
cd genai-mcp-server-advanced

# Install
uv sync

# Run MCP server (stdio)
uv run genai-mcp-advanced

# Run HTTP server (Grafana)
uv run genai-mcp-http-advanced
```

## 🛠️ MCP Tools

### Basic Tools
| Tool | Description |
|------|-------------|
| `get_summary` | Comprehensive metrics overview |
| `get_cost` | Cost breakdown |
| `get_tokens` | Token usage |
| `get_latency` | Latency percentiles |
| `get_quality` | Quality scores |
| `get_rag_metrics` | RAG precision/recall |
| `get_ttft` | Time to First Token |
| `get_security` | Security events |
| `get_errors` | Error counts |
| `get_requests` | Request counts |
| `get_applications` | List apps |
| `get_models` | List models |
| `execute_query` | Custom PromQL |

### Analytics Tools
| Tool | Description |
|------|-------------|
| `detect_anomalies` | Find unusual metric values using Z-score |
| `analyze_trends` | Identify trends with linear regression |
| `forecast_metric` | Predict future values with confidence intervals |
| `find_correlations` | Discover related metrics |
| `compare_periods` | Compare two time periods statistically |
| `calculate_percentiles` | Get p50/p90/p95/p99 values |

### Insights Tools
| Tool | Description |
|------|-------------|
| `get_recommendations` | Cost, quality, performance, security suggestions |
| `analyze_root_cause` | Identify probable causes for issues |
| `check_sla_compliance` | Monitor SLA status and predict breaches |
| `generate_health_report` | Full system health assessment |

## 💬 Natural Language Queries

### Anomaly Detection
```
"Detect anomalies in errors"
"Any unusual latency spikes?"
"Find outliers in cost"
```

### Trend Analysis
```
"Show me cost trends"
"Is error rate increasing?"
"Analyze request trends"
```

### Forecasting
```
"Forecast cost for next 24 hours"
"Predict future requests"
"What will latency be tomorrow?"
```

### Recommendations
```
"Give me cost optimization tips"
"How can I improve quality?"
"Recommend performance improvements"
"Any security recommendations?"
```

### Root Cause Analysis
```
"Why are errors spiking?"
"Diagnose latency issues"
"Investigate cost increase"
```

### SLA Monitoring
```
"Check SLA compliance"
"Any SLA breaches?"
"Show me SLA status"
```

### Health Report
```
"Show health report"
"System status"
"Give me an overview"
```

## 📊 Example Outputs

### Anomaly Detection
```
🚨 Anomaly Detection (genai_errors_total)

Found 2 anomalies:

- HIGH: genai_errors_total is 3.2σ above normal (45 vs mean 12)
- MEDIUM: genai_errors_total is 2.8σ above normal (38 vs mean 15)
```

### Trend Analysis
```
📈 Trend Analysis (genai_cost_dollars_total)

- Direction: INCREASING
- Change: +23.5%
- R²: 0.87
- Forecast (1h): $0.0234
- Forecast (24h): $0.1456
```

### Recommendations
```
💡 Cost Recommendations

🔴 Consider replacing gpt-4o with gpt-3.5-turbo
   gpt-4o costs $0.000045/request vs $0.000012/request
   💡 Evaluate if gpt-3.5-turbo can handle workloads

🟠 High error rate (8.2%) wasting budget
   523 failed requests out of 6384
   💡 Investigate and fix error causes
```

### SLA Compliance
```
📊 SLA Compliance (3/4 passing)

✅ Latency P95: 1234.00 (target: 3000) 📉
✅ Quality Score: 0.85 (target: 0.8) ➡️
❌ Error Rate: 2.30 (target: 1) ⚠️ ~4.2 hours 📈
✅ Availability: 97.70 (target: 99) 📉
```

## 🔧 Configuration

```bash
PROMETHEUS_URL=http://localhost:9090
MCP_PORT=3001
MCP_HOST=0.0.0.0
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              MCP Clients                    │
│  (Claude Desktop, Grafana, Custom Apps)     │
└─────────────────────┬───────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐         ┌─────────────────┐
│  MCP Server   │         │   HTTP Server   │
│   (FastMCP)   │         │    (Flask)      │
└───────┬───────┘         └────────┬────────┘
        │                          │
        └──────────┬───────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│ Tools  │   │Analytics │   │ Insights │
│(basic) │   │(ML/stats)│   │(smart)   │
└───┬────┘   └────┬─────┘   └────┬─────┘
    │             │              │
    └─────────────┴──────────────┘
                  │
                  ▼
         ┌───────────────┐
         │  Prometheus   │
         │   Client      │
         │  (cached)     │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │  Prometheus   │
         │    :9090      │
         └───────────────┘
```

## 📦 Dependencies

**Required:**
- fastmcp >= 2.0.0
- flask, flask-cors
- numpy, scipy
- pydantic
- cachetools

**Optional (ML features):**
- scikit-learn
- statsmodels

## 🧪 Development

```bash
# Install all dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

## 📄 License

MIT
