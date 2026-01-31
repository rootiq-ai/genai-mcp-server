"""Analytics: anomaly detection, trend analysis, forecasting."""
from dataclasses import dataclass
import numpy as np
from scipy import stats
from .prometheus import prom, parse_range_results, get_value


@dataclass
class AnomalyResult:
    metric_name: str
    labels: dict
    current_value: float
    mean: float
    std_dev: float
    z_score: float
    is_anomaly: bool
    severity: str
    direction: str
    message: str


@dataclass
class TrendResult:
    metric_name: str
    labels: dict
    direction: str
    slope: float
    r_squared: float
    change_percent: float
    forecast_1h: float
    forecast_24h: float
    message: str


@dataclass
class CorrelationResult:
    metric_a: str
    metric_b: str
    correlation: float
    p_value: float
    is_significant: bool
    relationship: str


def detect_anomalies(metric: str, labels: dict | None = None, lookback: str = "1h", z_threshold: float = 2.5) -> list[AnomalyResult]:
    """Detect anomalies using Z-score."""
    results = []
    label_str = ",".join([f'{k}="{v}"' for k, v in (labels or {}).items()])
    query = f"{metric}{{{label_str}}}" if label_str else metric
    
    range_resp = prom.query_range_relative(query, lookback, "60s")
    range_results = parse_range_results(range_resp)
    
    for series in range_results:
        if len(series.data_points) < 10:
            continue
        values = np.array(series.data_points)
        current = values[-1]
        mean = np.mean(values[:-1])
        std = np.std(values[:-1])
        if std == 0:
            continue
        z_score = (current - mean) / std
        is_anomaly = abs(z_score) > z_threshold
        
        if is_anomaly:
            direction = "above" if z_score > 0 else "below"
            if abs(z_score) > 4:
                severity = "critical"
            elif abs(z_score) > 3:
                severity = "high"
            elif abs(z_score) > 2.5:
                severity = "medium"
            else:
                severity = "low"
            
            results.append(AnomalyResult(
                metric_name=metric, labels=series.metric, current_value=current,
                mean=round(mean, 4), std_dev=round(std, 4), z_score=round(z_score, 2),
                is_anomaly=True, severity=severity, direction=direction,
                message=f"{metric} is {abs(z_score):.1f} std {direction} normal ({current:.4f} vs mean {mean:.4f})"
            ))
    return results


def analyze_trend(metric: str, labels: dict | None = None, lookback: str = "6h") -> list[TrendResult]:
    """Analyze trends using linear regression."""
    results = []
    label_str = ",".join([f'{k}="{v}"' for k, v in (labels or {}).items()])
    query = f"{metric}{{{label_str}}}" if label_str else metric
    
    range_resp = prom.query_range_relative(query, lookback, "60s")
    range_results = parse_range_results(range_resp)
    
    for series in range_results:
        if len(series.data_points) < 10:
            continue
        values = np.array(series.data_points)
        x = np.arange(len(values))
        slope, intercept, r_value, _, _ = stats.linregress(x, values)
        r_squared = r_value ** 2
        
        if abs(slope) < 0.001 * np.mean(values):
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        change_percent = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        forecast_1h = max(0, intercept + slope * (len(values) + 60))
        forecast_24h = max(0, intercept + slope * (len(values) + 1440))
        
        results.append(TrendResult(
            metric_name=metric, labels=series.metric, direction=direction,
            slope=round(slope, 6), r_squared=round(r_squared, 4),
            change_percent=round(change_percent, 2),
            forecast_1h=round(forecast_1h, 4), forecast_24h=round(forecast_24h, 4),
            message=f"{metric} is {direction} ({change_percent:+.1f}% change, R2={r_squared:.2f})"
        ))
    return results


def find_correlations(metrics: list[str] | None = None, lookback: str = "1h", min_correlation: float = 0.7) -> list[CorrelationResult]:
    """Find correlations between metrics."""
    if metrics is None:
        metrics = ["sum(genai_errors_total)", "avg(genai_latency_seconds)", "sum(genai_cost_dollars_total)", "sum(genai_requests_total)"]
    
    results = []
    metric_data = {}
    
    for metric in metrics:
        range_resp = prom.query_range_relative(metric, lookback, "60s")
        range_results = parse_range_results(range_resp)
        if range_results:
            metric_data[metric] = np.array(range_results[0].data_points)
    
    metric_list = list(metric_data.keys())
    for i, metric_a in enumerate(metric_list):
        for metric_b in metric_list[i + 1:]:
            data_a, data_b = metric_data[metric_a], metric_data[metric_b]
            min_len = min(len(data_a), len(data_b))
            if min_len < 10:
                continue
            data_a, data_b = data_a[-min_len:], data_b[-min_len:]
            correlation, p_value = stats.pearsonr(data_a, data_b)
            
            if abs(correlation) >= min_correlation:
                if correlation > 0.9:
                    relationship = "strong_positive"
                elif correlation > 0.7:
                    relationship = "moderate_positive"
                elif correlation > -0.7:
                    relationship = "weak"
                elif correlation > -0.9:
                    relationship = "moderate_negative"
                else:
                    relationship = "strong_negative"
                
                results.append(CorrelationResult(
                    metric_a=metric_a, metric_b=metric_b,
                    correlation=round(correlation, 4), p_value=round(p_value, 6),
                    is_significant=p_value < 0.05, relationship=relationship
                ))
    return results


def forecast_metric(metric: str, horizon_hours: int = 24, lookback: str = "24h") -> dict:
    """Forecast future values."""
    range_resp = prom.query_range_relative(metric, lookback, "300s")
    range_results = parse_range_results(range_resp)
    
    if not range_results or len(range_results[0].data_points) < 10:
        return {"error": "Insufficient data"}
    
    values = np.array(range_results[0].data_points)
    x = np.arange(len(values))
    slope, intercept, r_value, _, _ = stats.linregress(x, values)
    
    residuals = values - (intercept + slope * x)
    ci_95 = 1.96 * np.std(residuals)
    
    forecasts = []
    for h in range(1, min(horizon_hours + 1, 49)):
        idx = len(values) + h * 12
        pred = max(0, intercept + slope * idx)
        forecasts.append({
            "hours_ahead": h,
            "predicted": round(pred, 4),
            "lower_95": round(max(0, pred - ci_95), 4),
            "upper_95": round(pred + ci_95, 4)
        })
    
    return {
        "metric": metric,
        "current_value": round(values[-1], 4),
        "trend": {"slope": round(slope, 6), "r_squared": round(r_value**2, 4), "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"},
        "forecasts": forecasts
    }


def calculate_percentiles(metric: str, percentiles: list[float] | None = None, lookback: str = "1h") -> dict:
    """Calculate percentiles."""
    if percentiles is None:
        percentiles = [50, 90, 95, 99]
    
    range_resp = prom.query_range_relative(metric, lookback, "60s")
    range_results = parse_range_results(range_resp)
    
    if not range_results:
        return {"error": "No data"}
    
    all_values = []
    for series in range_results:
        all_values.extend(series.data_points)
    values = np.array(all_values)
    
    result = {"metric": metric, "lookback": lookback, "count": len(values), "percentiles": {}}
    for p in percentiles:
        result["percentiles"][f"p{int(p)}"] = round(np.percentile(values, p), 4)
    result["min"] = round(np.min(values), 4)
    result["max"] = round(np.max(values), 4)
    result["mean"] = round(np.mean(values), 4)
    return result
