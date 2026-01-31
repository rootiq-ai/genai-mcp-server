"""Chart generation module using Matplotlib."""
import base64
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from .prometheus import prom, parse_range_results, get_values
from . import tools, analytics, insights

logger = logging.getLogger(__name__)

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#34495e', '#e91e63']
BACKGROUND_COLOR = '#1e1e1e'
TEXT_COLOR = '#ffffff'
GRID_COLOR = '#333333'


def setup_dark_style():
    """Setup dark theme for charts."""
    plt.rcParams.update({
        'figure.facecolor': BACKGROUND_COLOR,
        'axes.facecolor': BACKGROUND_COLOR,
        'axes.edgecolor': GRID_COLOR,
        'axes.labelcolor': TEXT_COLOR,
        'text.color': TEXT_COLOR,
        'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR,
        'grid.color': GRID_COLOR,
        'legend.facecolor': BACKGROUND_COLOR,
        'legend.edgecolor': GRID_COLOR,
        'figure.figsize': (10, 6),
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor=BACKGROUND_COLOR, edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def create_line_chart(data: list[tuple[datetime, float]], title: str, ylabel: str, 
                      color: str = '#3498db') -> str:
    """Create a line chart and return as base64."""
    setup_dark_style()
    fig, ax = plt.subplots()
    
    if not data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                fontsize=14, color=TEXT_COLOR)
        ax.set_title(title)
        return fig_to_base64(fig)
    
    times, values = zip(*data)
    ax.plot(times, values, color=color, linewidth=2, marker='o', markersize=3)
    ax.fill_between(times, values, alpha=0.3, color=color)
    
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    # Add current value annotation
    if values:
        current = values[-1]
        ax.annotate(f'{current:.4f}', xy=(times[-1], current), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, color=color, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def create_multi_line_chart(data_series: dict[str, list[tuple[datetime, float]]], 
                            title: str, ylabel: str) -> str:
    """Create a multi-line chart and return as base64."""
    setup_dark_style()
    fig, ax = plt.subplots()
    
    if not data_series:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                fontsize=14, color=TEXT_COLOR)
        ax.set_title(title)
        return fig_to_base64(fig)
    
    for i, (label, data) in enumerate(data_series.items()):
        if data:
            times, values = zip(*data)
            color = COLORS[i % len(COLORS)]
            ax.plot(times, values, color=color, linewidth=2, label=label, 
                   marker='o', markersize=3)
    
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend(loc='upper left', framealpha=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig_to_base64(fig)


def create_bar_chart(labels: list[str], values: list[float], title: str, 
                     ylabel: str, horizontal: bool = False) -> str:
    """Create a bar chart and return as base64."""
    setup_dark_style()
    fig, ax = plt.subplots()
    
    if not labels or not values:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                fontsize=14, color=TEXT_COLOR)
        ax.set_title(title)
        return fig_to_base64(fig)
    
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    
    if horizontal:
        bars = ax.barh(labels, values, color=colors)
        ax.set_xlabel(ylabel)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}', 
                   va='center', color=TEXT_COLOR, fontsize=10)
    else:
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', 
                   ha='center', va='bottom', color=TEXT_COLOR, fontsize=10)
    
    ax.set_title(title, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig_to_base64(fig)


def create_pie_chart(labels: list[str], values: list[float], title: str) -> str:
    """Create a pie chart and return as base64."""
    setup_dark_style()
    fig, ax = plt.subplots()
    
    if not labels or not values or sum(values) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                fontsize=14, color=TEXT_COLOR)
        ax.set_title(title)
        return fig_to_base64(fig)
    
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'color': TEXT_COLOR})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig_to_base64(fig)


def create_gauge_chart(value: float, max_value: float, title: str, 
                       thresholds: tuple = (0.7, 0.9)) -> str:
    """Create a gauge chart and return as base64."""
    setup_dark_style()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # Gauge parameters
    start_angle = np.pi
    end_angle = 0
    
    # Draw background arc
    theta = np.linspace(start_angle, end_angle, 100)
    ax.fill_between(theta, 0.8, 1.0, color=GRID_COLOR, alpha=0.5)
    
    # Determine color based on value
    ratio = value / max_value if max_value > 0 else 0
    if ratio < thresholds[0]:
        color = '#2ecc71'  # Green
    elif ratio < thresholds[1]:
        color = '#f39c12'  # Yellow
    else:
        color = '#e74c3c'  # Red
    
    # Draw value arc
    value_angle = start_angle - (start_angle - end_angle) * min(ratio, 1.0)
    theta_value = np.linspace(start_angle, value_angle, 100)
    ax.fill_between(theta_value, 0.8, 1.0, color=color)
    
    # Add center text
    ax.text(0, 0, f'{value:.2f}', ha='center', va='center', 
            fontsize=24, fontweight='bold', color=TEXT_COLOR)
    ax.text(0, -0.3, f'/ {max_value:.2f}', ha='center', va='center', 
            fontsize=12, color=TEXT_COLOR, alpha=0.7)
    
    ax.set_ylim(0, 1)
    ax.set_title(title, fontweight='bold', pad=20, y=1.1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def create_heatmap(data: list[list[float]], x_labels: list[str], 
                   y_labels: list[str], title: str) -> str:
    """Create a heatmap and return as base64."""
    setup_dark_style()
    fig, ax = plt.subplots()
    
    if not data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                fontsize=14, color=TEXT_COLOR)
        ax.set_title(title)
        return fig_to_base64(fig)
    
    data_array = np.array(data)
    im = ax.imshow(data_array, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.xticks(rotation=45, ha='right')
    
    # Add text annotations
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, f'{data_array[i, j]:.2f}',
                          ha='center', va='center', color='white', fontsize=9)
    
    ax.set_title(title, fontweight='bold', pad=20)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# SPECIFIC CHART GENERATORS
# =============================================================================

def chart_cost_trend(lookback: str = "6h") -> str:
    """Generate cost trend chart."""
    range_resp = prom.query_range_relative("sum(genai_cost_dollars_total)", lookback, "60s")
    results = parse_range_results(range_resp)
    
    data = []
    if results:
        for ts, val in results[0].values:
            data.append((datetime.fromtimestamp(ts), val))
    
    return create_line_chart(data, "💰 Cost Trend", "Cost (USD)", color='#2ecc71')


def chart_requests_trend(lookback: str = "6h") -> str:
    """Generate requests trend chart."""
    range_resp = prom.query_range_relative("sum(rate(genai_requests_total[5m]))", lookback, "60s")
    results = parse_range_results(range_resp)
    
    data = []
    if results:
        for ts, val in results[0].values:
            data.append((datetime.fromtimestamp(ts), val))
    
    return create_line_chart(data, "📈 Request Rate Trend", "Requests/sec", color='#3498db')


def chart_latency_trend(lookback: str = "6h") -> str:
    """Generate latency trend chart."""
    range_resp = prom.query_range_relative(
        'avg(genai_latency_seconds{quantile="0.95"}) * 1000', lookback, "60s"
    )
    results = parse_range_results(range_resp)
    
    data = []
    if results:
        for ts, val in results[0].values:
            data.append((datetime.fromtimestamp(ts), val))
    
    return create_line_chart(data, "⚡ P95 Latency Trend", "Latency (ms)", color='#e74c3c')


def chart_errors_trend(lookback: str = "6h") -> str:
    """Generate errors trend chart."""
    range_resp = prom.query_range_relative("sum(rate(genai_errors_total[5m]))", lookback, "60s")
    results = parse_range_results(range_resp)
    
    data = []
    if results:
        for ts, val in results[0].values:
            data.append((datetime.fromtimestamp(ts), val))
    
    return create_line_chart(data, "❌ Error Rate Trend", "Errors/sec", color='#e74c3c')


def chart_tokens_trend(lookback: str = "6h") -> str:
    """Generate tokens trend chart."""
    input_resp = prom.query_range_relative("sum(genai_input_tokens_total)", lookback, "60s")
    output_resp = prom.query_range_relative("sum(genai_output_tokens_total)", lookback, "60s")
    
    input_results = parse_range_results(input_resp)
    output_results = parse_range_results(output_resp)
    
    data_series = {}
    if input_results:
        data_series['Input Tokens'] = [(datetime.fromtimestamp(ts), val) 
                                        for ts, val in input_results[0].values]
    if output_results:
        data_series['Output Tokens'] = [(datetime.fromtimestamp(ts), val) 
                                         for ts, val in output_results[0].values]
    
    return create_multi_line_chart(data_series, "📊 Token Usage Trend", "Tokens")


def chart_cost_by_model() -> str:
    """Generate cost breakdown by model chart."""
    cost_data = tools.get_cost()
    
    labels = [item.get('model', 'unknown') for item in cost_data['breakdown']]
    values = [item['cost_usd'] for item in cost_data['breakdown']]
    
    if sum(values) > 0:
        return create_pie_chart(labels, values, "💰 Cost by Model")
    return create_bar_chart(labels, values, "💰 Cost by Model", "Cost (USD)")


def chart_requests_by_model() -> str:
    """Generate requests breakdown by model chart."""
    req_data = tools.get_requests()
    
    labels = [item.get('model', 'unknown') for item in req_data['requests']]
    values = [item['count'] for item in req_data['requests']]
    
    return create_bar_chart(labels, values, "📈 Requests by Model", "Request Count")


def chart_latency_by_model() -> str:
    """Generate latency comparison by model chart."""
    lat_data = tools.get_latency()
    
    if not lat_data['latency_ms']:
        setup_dark_style()
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No latency data available', ha='center', va='center', 
                fontsize=14, color=TEXT_COLOR)
        ax.set_title("⚡ Latency by Model")
        return fig_to_base64(fig)
    
    labels = [item.get('model', 'unknown') for item in lat_data['latency_ms']]
    p50_values = [item.get('p05', 0) for item in lat_data['latency_ms']]
    p95_values = [item.get('p095', 0) for item in lat_data['latency_ms']]
    p99_values = [item.get('p099', 0) for item in lat_data['latency_ms']]
    
    setup_dark_style()
    fig, ax = plt.subplots()
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, p50_values, width, label='P50', color=COLORS[0])
    bars2 = ax.bar(x, p95_values, width, label='P95', color=COLORS[1])
    bars3 = ax.bar(x + width, p99_values, width, label='P99', color=COLORS[2])
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('⚡ Latency by Model', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)


def chart_quality_by_app() -> str:
    """Generate quality scores by application chart."""
    qual_data = tools.get_quality()
    
    if not qual_data['quality']:
        setup_dark_style()
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No quality data available', ha='center', va='center', 
                fontsize=14, color=TEXT_COLOR)
        ax.set_title("✅ Quality by Application")
        return fig_to_base64(fig)
    
    labels = [item.get('application', 'unknown') for item in qual_data['quality']]
    groundedness = [item.get('groundedness', 0) * 100 for item in qual_data['quality']]
    relevance = [item.get('relevance', 0) * 100 for item in qual_data['quality']]
    coherence = [item.get('coherence', 0) * 100 for item in qual_data['quality']]
    fluency = [item.get('fluency', 0) * 100 for item in qual_data['quality']]
    
    setup_dark_style()
    fig, ax = plt.subplots()
    
    x = np.arange(len(labels))
    width = 0.2
    
    ax.bar(x - 1.5*width, groundedness, width, label='Groundedness', color=COLORS[0])
    ax.bar(x - 0.5*width, relevance, width, label='Relevance', color=COLORS[1])
    ax.bar(x + 0.5*width, coherence, width, label='Coherence', color=COLORS[2])
    ax.bar(x + 1.5*width, fluency, width, label='Fluency', color=COLORS[3])
    
    ax.set_ylabel('Score (%)')
    ax.set_title('✅ Quality Scores by Application', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def chart_errors_by_type() -> str:
    """Generate errors by type chart."""
    err_data = tools.get_errors()
    
    if not err_data['errors']:
        labels = ['No Errors']
        values = [0]
    else:
        labels = [item.get('error_type', 'unknown') for item in err_data['errors']]
        values = [item['count'] for item in err_data['errors']]
    
    if sum(values) > 0:
        return create_pie_chart(labels, values, "❌ Errors by Type")
    return create_bar_chart(labels, values, "❌ Errors by Type", "Error Count")


def chart_security_events() -> str:
    """Generate security events chart."""
    sec_data = tools.get_security()
    
    labels = [item.get('type', 'unknown') for item in sec_data['guardrail_triggers']]
    values = [item['count'] for item in sec_data['guardrail_triggers']]
    
    if not labels:
        labels = ['No Events']
        values = [0]
    
    return create_bar_chart(labels, values, "🔒 Security Events", "Event Count", horizontal=True)


def chart_health_gauge() -> str:
    """Generate health score gauge."""
    health = insights.generate_health_report()
    summary = health['summary']
    
    # Calculate health score (0-100)
    score = 100
    if summary['error_rate_percent'] > 5:
        score -= 30
    elif summary['error_rate_percent'] > 1:
        score -= 15
    
    if summary['avg_latency_ms'] > 3000:
        score -= 25
    elif summary['avg_latency_ms'] > 1000:
        score -= 10
    
    if summary['avg_quality'] < 0.7:
        score -= 25
    elif summary['avg_quality'] < 0.8:
        score -= 10
    
    score = max(0, score)
    
    return create_gauge_chart(score, 100, "🏥 System Health Score", thresholds=(70, 90))


def chart_forecast(metric: str = "genai_cost_dollars_total", hours: int = 24) -> str:
    """Generate forecast chart."""
    forecast_data = analytics.forecast_metric(metric, hours, "24h")
    
    if "error" in forecast_data:
        setup_dark_style()
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Cannot forecast: {forecast_data["error"]}', 
                ha='center', va='center', fontsize=14, color=TEXT_COLOR)
        ax.set_title("🔮 Forecast")
        return fig_to_base64(fig)
    
    setup_dark_style()
    fig, ax = plt.subplots()
    
    # Plot forecasts
    hours_ahead = [f['hours_ahead'] for f in forecast_data['forecasts']]
    predicted = [f['predicted'] for f in forecast_data['forecasts']]
    lower_95 = [f['lower_95'] for f in forecast_data['forecasts']]
    upper_95 = [f['upper_95'] for f in forecast_data['forecasts']]
    
    ax.plot(hours_ahead, predicted, color='#3498db', linewidth=2, marker='o', 
            markersize=4, label='Predicted')
    ax.fill_between(hours_ahead, lower_95, upper_95, alpha=0.3, color='#3498db', 
                    label='95% CI')
    
    # Add current value
    ax.axhline(y=forecast_data['current_value'], color='#2ecc71', linestyle='--', 
               linewidth=1, label=f'Current: {forecast_data["current_value"]:.4f}')
    
    ax.set_xlabel('Hours Ahead')
    ax.set_ylabel('Value')
    ax.set_title(f'🔮 Forecast: {metric}', fontweight='bold', pad=20)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def chart_anomalies() -> str:
    """Generate anomaly visualization chart."""
    setup_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ("genai_errors_total", "Errors", '#e74c3c'),
        ("genai_latency_seconds", "Latency", '#f39c12'),
        ("genai_cost_dollars_total", "Cost", '#2ecc71')
    ]
    
    for ax, (metric, label, color) in zip(axes, metrics):
        # Get historical data
        range_resp = prom.query_range_relative(metric, "1h", "60s")
        results = parse_range_results(range_resp)
        
        if results and results[0].values:
            times = [datetime.fromtimestamp(ts) for ts, _ in results[0].values]
            values = [val for _, val in results[0].values]
            
            ax.plot(times, values, color=color, linewidth=1.5)
            
            # Detect and highlight anomalies
            anomalies = analytics.detect_anomalies(metric, lookback="1h")
            if anomalies:
                ax.scatter([times[-1]], [values[-1]], color='red', s=100, 
                          zorder=5, marker='!', label='Anomaly')
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   fontsize=12, color=TEXT_COLOR)
        
        ax.set_title(label, fontweight='bold')
    
    fig.suptitle('🚨 Anomaly Detection', fontweight='bold', fontsize=14)
    plt.tight_layout()
    return fig_to_base64(fig)


def chart_trends() -> str:
    """Generate trend analysis chart."""
    setup_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = [
        ("sum(genai_requests_total)", "Requests", '#3498db'),
        ("sum(genai_cost_dollars_total)", "Cost", '#2ecc71'),
        ("sum(genai_errors_total)", "Errors", '#e74c3c'),
        ('avg(genai_latency_seconds{quantile="0.95"})*1000', "P95 Latency (ms)", '#f39c12')
    ]
    
    for ax, (metric, label, color) in zip(axes, metrics):
        range_resp = prom.query_range_relative(metric, "6h", "60s")
        results = parse_range_results(range_resp)
        
        if results and results[0].values:
            times = [datetime.fromtimestamp(ts) for ts, _ in results[0].values]
            values = [val for _, val in results[0].values]
            
            ax.plot(times, values, color=color, linewidth=2)
            ax.fill_between(times, values, alpha=0.3, color=color)
            
            # Add trend line
            if len(values) > 10:
                x = np.arange(len(values))
                z = np.polyfit(x, values, 1)
                p = np.poly1d(z)
                ax.plot(times, p(x), '--', color='white', alpha=0.7, linewidth=1)
                
                trend_dir = "📈" if z[0] > 0 else "📉" if z[0] < 0 else "➡️"
                ax.set_title(f'{label} {trend_dir}', fontweight='bold')
            else:
                ax.set_title(label, fontweight='bold')
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   fontsize=12, color=TEXT_COLOR)
            ax.set_title(label, fontweight='bold')
    
    fig.suptitle('📈 Trend Analysis (6 hours)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    return fig_to_base64(fig)


def chart_dashboard() -> str:
    """Generate a comprehensive dashboard chart."""
    setup_dark_style()
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    summary = tools.get_summary()['summary']
    
    # 1. Health Status (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    health = insights.generate_health_report()
    status = health['health_status']
    color = '#2ecc71' if status == 'healthy' else '#f39c12' if status == 'warning' else '#e74c3c'
    ax1.pie([1], colors=[color], radius=0.8)
    ax1.text(0, 0, status.upper(), ha='center', va='center', 
             fontsize=16, fontweight='bold', color='white')
    ax1.set_title('System Health', fontweight='bold')
    
    # 2. Key Metrics (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_text = f"""
    Requests: {summary['total_requests']:,}
    Cost: ${summary['total_cost_usd']:.4f}
    Tokens: {summary['total_tokens']:,}
    Error Rate: {summary['error_rate_percent']:.1f}%
    Avg Latency: {summary['avg_latency_ms']:.0f}ms
    Quality: {summary['avg_quality']*100:.1f}%
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center', color=TEXT_COLOR)
    ax2.axis('off')
    ax2.set_title('Key Metrics', fontweight='bold')
    
    # 3. Cost by Model (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    cost_data = tools.get_cost()
    if cost_data['breakdown']:
        labels = [item.get('model', '?')[:10] for item in cost_data['breakdown']]
        values = [item['cost_usd'] for item in cost_data['breakdown']]
        colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
        ax3.pie(values, labels=labels, colors=colors, autopct='%1.0f%%', 
               textprops={'fontsize': 8, 'color': 'white'})
    ax3.set_title('Cost by Model', fontweight='bold')
    
    # 4. Request Trend (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    range_resp = prom.query_range_relative("sum(rate(genai_requests_total[5m]))", "6h", "120s")
    results = parse_range_results(range_resp)
    if results and results[0].values:
        times = [datetime.fromtimestamp(ts) for ts, _ in results[0].values]
        values = [val for _, val in results[0].values]
        ax4.plot(times, values, color='#3498db', linewidth=2)
        ax4.fill_between(times, values, alpha=0.3, color='#3498db')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.set_title('Request Rate', fontweight='bold')
    
    # 5. Latency Trend (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    range_resp = prom.query_range_relative('avg(genai_latency_seconds{quantile="0.95"})*1000', "6h", "120s")
    results = parse_range_results(range_resp)
    if results and results[0].values:
        times = [datetime.fromtimestamp(ts) for ts, _ in results[0].values]
        values = [val for _, val in results[0].values]
        ax5.plot(times, values, color='#e74c3c', linewidth=2)
        ax5.fill_between(times, values, alpha=0.3, color='#e74c3c')
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax5.set_title('P95 Latency (ms)', fontweight='bold')
    
    # 6. Error Trend (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    range_resp = prom.query_range_relative("sum(rate(genai_errors_total[5m]))", "6h", "120s")
    results = parse_range_results(range_resp)
    if results and results[0].values:
        times = [datetime.fromtimestamp(ts) for ts, _ in results[0].values]
        values = [val for _, val in results[0].values]
        ax6.plot(times, values, color='#f39c12', linewidth=2)
        ax6.fill_between(times, values, alpha=0.3, color='#f39c12')
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax6.set_title('Error Rate', fontweight='bold')
    
    # 7. Quality by App (bottom, spans 2 columns)
    ax7 = fig.add_subplot(gs[2, 0:2])
    qual_data = tools.get_quality()
    if qual_data['quality']:
        apps = [q['application'][:15] for q in qual_data['quality']]
        scores = [q.get('overall', 0) * 100 for q in qual_data['quality']]
        colors = [COLORS[i % len(COLORS)] for i in range(len(apps))]
        bars = ax7.barh(apps, scores, color=colors)
        ax7.set_xlim(0, 100)
        ax7.axvline(x=80, color='#2ecc71', linestyle='--', alpha=0.5)
        for bar, score in zip(bars, scores):
            ax7.text(score + 1, bar.get_y() + bar.get_height()/2, 
                    f'{score:.0f}%', va='center', fontsize=9, color=TEXT_COLOR)
    ax7.set_title('Quality by Application (%)', fontweight='bold')
    
    # 8. Security Events (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    sec_data = tools.get_security()
    if sec_data['guardrail_triggers']:
        types = [t['type'][:12] for t in sec_data['guardrail_triggers']]
        counts = [t['count'] for t in sec_data['guardrail_triggers']]
        colors = [COLORS[i % len(COLORS)] for i in range(len(types))]
        ax8.bar(types, counts, color=colors)
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax8.text(0.5, 0.5, 'No security events', ha='center', va='center', 
                fontsize=12, color='#2ecc71')
    ax8.set_title('Security Events', fontweight='bold')
    
    fig.suptitle('📊 GenAI Observability Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# CHART ROUTER
# =============================================================================

CHART_FUNCTIONS = {
    'cost_trend': chart_cost_trend,
    'cost_by_model': chart_cost_by_model,
    'requests_trend': chart_requests_trend,
    'requests_by_model': chart_requests_by_model,
    'latency_trend': chart_latency_trend,
    'latency_by_model': chart_latency_by_model,
    'tokens_trend': chart_tokens_trend,
    'errors_trend': chart_errors_trend,
    'errors_by_type': chart_errors_by_type,
    'quality_by_app': chart_quality_by_app,
    'security_events': chart_security_events,
    'health_gauge': chart_health_gauge,
    'forecast': chart_forecast,
    'anomalies': chart_anomalies,
    'trends': chart_trends,
    'dashboard': chart_dashboard,
}


def generate_chart(chart_type: str, **kwargs) -> Optional[str]:
    """Generate a chart by type and return base64 PNG."""
    if chart_type in CHART_FUNCTIONS:
        try:
            return CHART_FUNCTIONS[chart_type](**kwargs)
        except Exception as e:
            logger.error(f"Error generating chart {chart_type}: {e}")
            return None
    return None


def list_available_charts() -> list[str]:
    """Return list of available chart types."""
    return list(CHART_FUNCTIONS.keys())
