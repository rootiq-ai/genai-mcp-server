"""Advanced analytics: anomaly detection, trend analysis, forecasting, correlations."""
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy import stats

from .prometheus import prom, parse_range_results, get_value

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    metric_name: str
    labels: dict[str, str]
    current_value: float
    mean: float
    std_dev: float
    z_score: float
    is_anomaly: bool
    severity: str  # low, medium, high, critical
    direction: str  # above, below
    message: str


@dataclass
class TrendResult:
    """Trend analysis result."""
    metric_name: str
    labels: dict[str, str]
    direction: str  # increasing, decreasing, stable
    slope: float
    r_squared: float
    change_percent: float
    forecast_1h: float
    forecast_24h: float
    message: str


@dataclass
class CorrelationResult:
    """Correlation analysis result."""
    metric_a: str
    metric_b: str
    correlation: float
    p_value: float
    is_significant: bool
    relationship: str  # strong_positive, moderate_positive, weak, moderate_negative, strong_negative


def detect_anomalies(
    metric: str,
    labels: dict[str, str] | None = None,
    lookback: str = "1h",
    z_threshold: float = 2.5,
) -> list[AnomalyResult]:
    """
    Detect anomalies using Z-score method.
    
    Args:
        metric: Metric name or PromQL expression
        labels: Optional label filters
        lookback: Time range for baseline (e.g., '1h', '6h', '24h')
        z_threshold: Z-score threshold for anomaly (default 2.5)
    
    Returns:
        List of detected anomalies
    """
    results = []
    
    # Build query
    label_str = ",".join([f'{k}="{v}"' for k, v in (labels or {}).items()])
    query = f"{metric}{{{label_str}}}" if label_str else metric
    
    # Get range data
    range_resp = prom.query_range_relative(query, lookback, "60s")
    range_results = parse_range_results(range_resp)
    
    for series in range_results:
        if len(series.data_points) < 10:
            continue
        
        values = np.array(series.data_points)
        current = values[-1]
        mean = np.mean(values[:-1])  # Exclude current for baseline
        std = np.std(values[:-1])
        
        if std == 0:
            continue
        
        z_score = (current - mean) / std
        is_anomaly = abs(z_score) > z_threshold
        
        if is_anomaly:
            direction = "above" if z_score > 0 else "below"
            
            # Determine severity
            if abs(z_score) > 4:
                severity = "critical"
            elif abs(z_score) > 3:
                severity = "high"
            elif abs(z_score) > 2.5:
                severity = "medium"
            else:
                severity = "low"
            
            results.append(AnomalyResult(
                metric_name=metric,
                labels=series.metric,
                current_value=current,
                mean=round(mean, 4),
                std_dev=round(std, 4),
                z_score=round(z_score, 2),
                is_anomaly=True,
                severity=severity,
                direction=direction,
                message=f"{metric} is {abs(z_score):.1f}σ {direction} normal ({current:.4f} vs mean {mean:.4f})",
            ))
    
    return results


def analyze_trend(
    metric: str,
    labels: dict[str, str] | None = None,
    lookback: str = "6h",
) -> list[TrendResult]:
    """
    Analyze metric trends using linear regression.
    
    Args:
        metric: Metric name or PromQL expression
        labels: Optional label filters
        lookback: Time range for analysis
    
    Returns:
        List of trend analysis results
    """
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
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        r_squared = r_value ** 2
        
        # Determine trend direction
        if abs(slope) < 0.001 * np.mean(values):
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Calculate change percentage
        if values[0] != 0:
            change_percent = ((values[-1] - values[0]) / values[0]) * 100
        else:
            change_percent = 0
        
        # Forecast
        points_per_hour = 60  # assuming 60s step
        forecast_1h = intercept + slope * (len(values) + points_per_hour)
        forecast_24h = intercept + slope * (len(values) + 24 * points_per_hour)
        
        results.append(TrendResult(
            metric_name=metric,
            labels=series.metric,
            direction=direction,
            slope=round(slope, 6),
            r_squared=round(r_squared, 4),
            change_percent=round(change_percent, 2),
            forecast_1h=round(max(0, forecast_1h), 4),
            forecast_24h=round(max(0, forecast_24h), 4),
            message=f"{metric} is {direction} ({change_percent:+.1f}% change, R²={r_squared:.2f})",
        ))
    
    return results


def find_correlations(
    metrics: list[str],
    lookback: str = "1h",
    min_correlation: float = 0.7,
) -> list[CorrelationResult]:
    """
    Find correlations between metrics.
    
    Args:
        metrics: List of metrics to compare
        lookback: Time range for analysis
        min_correlation: Minimum absolute correlation to report
    
    Returns:
        List of significant correlations
    """
    results = []
    
    # Get data for all metrics
    metric_data = {}
    for metric in metrics:
        range_resp = prom.query_range_relative(metric, lookback, "60s")
        range_results = parse_range_results(range_resp)
        if range_results:
            metric_data[metric] = np.array(range_results[0].data_points)
    
    # Compare pairs
    metric_list = list(metric_data.keys())
    for i, metric_a in enumerate(metric_list):
        for metric_b in metric_list[i + 1:]:
            data_a = metric_data[metric_a]
            data_b = metric_data[metric_b]
            
            # Align lengths
            min_len = min(len(data_a), len(data_b))
            if min_len < 10:
                continue
            
            data_a = data_a[-min_len:]
            data_b = data_b[-min_len:]
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(data_a, data_b)
            
            if abs(correlation) >= min_correlation:
                # Determine relationship strength
                if correlation > 0.9:
                    relationship = "strong_positive"
                elif correlation > 0.7:
                    relationship = "moderate_positive"
                elif correlation > 0.3:
                    relationship = "weak_positive"
                elif correlation > -0.3:
                    relationship = "weak"
                elif correlation > -0.7:
                    relationship = "moderate_negative"
                else:
                    relationship = "strong_negative"
                
                results.append(CorrelationResult(
                    metric_a=metric_a,
                    metric_b=metric_b,
                    correlation=round(correlation, 4),
                    p_value=round(p_value, 6),
                    is_significant=p_value < 0.05,
                    relationship=relationship,
                ))
    
    return results


def compare_periods(
    metric: str,
    period_a: tuple[str, str],  # (start, end) ISO format
    period_b: tuple[str, str],
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Compare metric values between two time periods.
    
    Args:
        metric: Metric name
        period_a: First period (start, end)
        period_b: Second period (start, end)
        labels: Optional label filters
    
    Returns:
        Comparison statistics
    """
    label_str = ",".join([f'{k}="{v}"' for k, v in (labels or {}).items()])
    query = f"{metric}{{{label_str}}}" if label_str else metric
    
    # Get data for both periods
    resp_a = prom.query_range(query, period_a[0], period_a[1], "60s")
    resp_b = prom.query_range(query, period_b[0], period_b[1], "60s")
    
    results_a = parse_range_results(resp_a)
    results_b = parse_range_results(resp_b)
    
    if not results_a or not results_b:
        return {"error": "Insufficient data for comparison"}
    
    values_a = np.array(results_a[0].data_points)
    values_b = np.array(results_b[0].data_points)
    
    # Statistics
    stats_a = {
        "mean": round(np.mean(values_a), 4),
        "min": round(np.min(values_a), 4),
        "max": round(np.max(values_a), 4),
        "std": round(np.std(values_a), 4),
        "median": round(np.median(values_a), 4),
    }
    
    stats_b = {
        "mean": round(np.mean(values_b), 4),
        "min": round(np.min(values_b), 4),
        "max": round(np.max(values_b), 4),
        "std": round(np.std(values_b), 4),
        "median": round(np.median(values_b), 4),
    }
    
    # Changes
    mean_change = stats_b["mean"] - stats_a["mean"]
    if stats_a["mean"] != 0:
        mean_change_pct = (mean_change / stats_a["mean"]) * 100
    else:
        mean_change_pct = 0
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)
    
    return {
        "metric": metric,
        "period_a": {"start": period_a[0], "end": period_a[1], "stats": stats_a},
        "period_b": {"start": period_b[0], "end": period_b[1], "stats": stats_b},
        "comparison": {
            "mean_change": round(mean_change, 4),
            "mean_change_percent": round(mean_change_pct, 2),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "is_significant": p_value < 0.05,
            "direction": "increased" if mean_change > 0 else "decreased" if mean_change < 0 else "unchanged",
        },
    }


def calculate_percentiles(
    metric: str,
    percentiles: list[float] = [50, 90, 95, 99],
    lookback: str = "1h",
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Calculate percentiles for a metric over a time range."""
    label_str = ",".join([f'{k}="{v}"' for k, v in (labels or {}).items()])
    query = f"{metric}{{{label_str}}}" if label_str else metric
    
    range_resp = prom.query_range_relative(query, lookback, "60s")
    range_results = parse_range_results(range_resp)
    
    if not range_results:
        return {"error": "No data"}
    
    all_values = []
    for series in range_results:
        all_values.extend(series.data_points)
    
    values = np.array(all_values)
    
    result = {
        "metric": metric,
        "lookback": lookback,
        "count": len(values),
        "percentiles": {},
    }
    
    for p in percentiles:
        result["percentiles"][f"p{int(p)}"] = round(np.percentile(values, p), 4)
    
    result["min"] = round(np.min(values), 4)
    result["max"] = round(np.max(values), 4)
    result["mean"] = round(np.mean(values), 4)
    result["std"] = round(np.std(values), 4)
    
    return result


def forecast_metric(
    metric: str,
    horizon_hours: int = 24,
    lookback: str = "24h",
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Simple linear forecast for a metric.
    
    Args:
        metric: Metric to forecast
        horizon_hours: Hours to forecast ahead
        lookback: Historical data to use
        labels: Optional label filters
    
    Returns:
        Forecast with confidence intervals
    """
    label_str = ",".join([f'{k}="{v}"' for k, v in (labels or {}).items()])
    query = f"{metric}{{{label_str}}}" if label_str else metric
    
    range_resp = prom.query_range_relative(query, lookback, "300s")  # 5min intervals
    range_results = parse_range_results(range_resp)
    
    if not range_results or len(range_results[0].data_points) < 10:
        return {"error": "Insufficient data for forecasting"}
    
    values = np.array(range_results[0].data_points)
    x = np.arange(len(values))
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    
    # Forecast points (5min intervals)
    forecast_points = horizon_hours * 12
    future_x = np.arange(len(values), len(values) + forecast_points)
    forecast_values = intercept + slope * future_x
    
    # Confidence interval (95%)
    residuals = values - (intercept + slope * x)
    residual_std = np.std(residuals)
    ci_95 = 1.96 * residual_std
    
    # Create hourly forecasts
    forecasts = []
    for h in range(1, horizon_hours + 1):
        idx = h * 12 - 1  # 5min intervals
        if idx < len(forecast_values):
            forecasts.append({
                "hours_ahead": h,
                "predicted": round(max(0, forecast_values[idx]), 4),
                "lower_95": round(max(0, forecast_values[idx] - ci_95), 4),
                "upper_95": round(forecast_values[idx] + ci_95, 4),
            })
    
    return {
        "metric": metric,
        "current_value": round(values[-1], 4),
        "trend": {
            "slope": round(slope, 6),
            "r_squared": round(r_value ** 2, 4),
            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
        },
        "forecasts": forecasts,
        "confidence_level": 0.95,
    }
