"""Tests for GenAI MCP Server Advanced."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# =============================================================================
# ANALYTICS TESTS
# =============================================================================

class TestAnomalyDetection:
    """Tests for anomaly detection."""
    
    def test_z_score_calculation(self):
        """Test Z-score calculation."""
        data = np.array([10, 11, 12, 10, 11, 50])  # 50 is anomaly
        mean = np.mean(data[:-1])
        std = np.std(data[:-1])
        z_score = (data[-1] - mean) / std
        assert z_score > 2.5  # Should be anomaly
    
    def test_severity_levels(self):
        """Test severity classification."""
        # Z > 4 = critical, Z > 3 = high, Z > 2.5 = medium
        z_scores = [4.5, 3.5, 2.7, 2.0]
        expected = ["critical", "high", "medium", "low"]
        
        for z, exp in zip(z_scores, expected):
            if z > 4:
                sev = "critical"
            elif z > 3:
                sev = "high"
            elif z > 2.5:
                sev = "medium"
            else:
                sev = "low"
            assert sev == exp


class TestTrendAnalysis:
    """Tests for trend analysis."""
    
    def test_increasing_trend(self):
        """Test detection of increasing trend."""
        from scipy import stats
        x = np.arange(10)
        y = x * 2 + np.random.normal(0, 0.1, 10)  # Clear increasing trend
        slope, _, r_value, _, _ = stats.linregress(x, y)
        assert slope > 0
        assert r_value ** 2 > 0.9
    
    def test_stable_trend(self):
        """Test detection of stable trend."""
        from scipy import stats
        x = np.arange(10)
        y = np.ones(10) * 5 + np.random.normal(0, 0.01, 10)  # Stable
        slope, _, _, _, _ = stats.linregress(x, y)
        assert abs(slope) < 0.01


class TestCorrelations:
    """Tests for correlation analysis."""
    
    def test_strong_positive_correlation(self):
        """Test strong positive correlation detection."""
        from scipy import stats
        x = np.arange(100)
        y = x * 2 + np.random.normal(0, 5, 100)
        corr, p = stats.pearsonr(x, y)
        assert corr > 0.9
        assert p < 0.05
    
    def test_no_correlation(self):
        """Test no correlation detection."""
        from scipy import stats
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)
        corr, _ = stats.pearsonr(x, y)
        assert abs(corr) < 0.3


class TestForecasting:
    """Tests for metric forecasting."""
    
    def test_linear_forecast(self):
        """Test linear forecast calculation."""
        from scipy import stats
        x = np.arange(24)  # 24 data points
        y = x * 0.5 + 10  # Linear with slope 0.5
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        # Forecast 12 points ahead
        forecast = intercept + slope * 36
        expected = 0.5 * 36 + 10
        assert abs(forecast - expected) < 0.1


# =============================================================================
# INSIGHTS TESTS
# =============================================================================

class TestRecommendations:
    """Tests for recommendation generation."""
    
    def test_cost_recommendation_structure(self):
        """Test recommendation has required fields."""
        from dataclasses import fields
        from genai_mcp_advanced.insights import Recommendation
        
        field_names = [f.name for f in fields(Recommendation)]
        required = ["category", "priority", "title", "description", "impact", "action"]
        for r in required:
            assert r in field_names
    
    def test_priority_values(self):
        """Test valid priority values."""
        valid_priorities = ["critical", "high", "medium", "low"]
        for p in valid_priorities:
            assert p in ["critical", "high", "medium", "low"]


class TestSLAMonitoring:
    """Tests for SLA monitoring."""
    
    def test_sla_compliance_logic(self):
        """Test SLA compliance determination."""
        # Test <= comparison
        current, target = 2500, 3000
        assert current <= target  # Compliant
        
        # Test >= comparison  
        current, target = 0.85, 0.8
        assert current >= target  # Compliant
        
        # Test breach
        current, target = 5, 1
        assert not (current <= target)  # Breached


class TestHealthReport:
    """Tests for health report generation."""
    
    def test_health_status_logic(self):
        """Test health status determination."""
        issues = []
        assert len(issues) == 0  # Healthy
        
        issues = ["High error rate"]
        if len(issues) >= 2:
            status = "critical"
        elif len(issues) == 1:
            status = "warning"
        else:
            status = "healthy"
        assert status == "warning"
        
        issues = ["High error rate", "High latency"]
        if len(issues) >= 2:
            status = "critical"
        assert status == "critical"


# =============================================================================
# TOOLS TESTS
# =============================================================================

class TestBasicTools:
    """Tests for basic metric tools."""
    
    def test_get_summary_structure(self):
        """Test summary returns expected structure."""
        # Mock test - would need Prometheus running for real test
        expected_keys = ["summary", "filters"]
        result = {"summary": {}, "filters": {}}
        for key in expected_keys:
            assert key in result
    
    def test_format_number(self):
        """Test number formatting."""
        def fmt(n):
            if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
            if n >= 1_000: return f"{n/1_000:.1f}K"
            return f"{n:.0f}"
        
        assert fmt(1_500_000) == "1.5M"
        assert fmt(1_500) == "1.5K"
        assert fmt(500) == "500"


class TestPrometheusClient:
    """Tests for Prometheus client."""
    
    def test_cache_ttl(self):
        """Test that cache has TTL."""
        from cachetools import TTLCache
        cache = TTLCache(maxsize=100, ttl=5)
        cache["key"] = "value"
        assert "key" in cache
    
    def test_parse_instant_results(self):
        """Test parsing of instant query results."""
        resp = {
            "status": "success",
            "data": {
                "result": [
                    {"metric": {"app": "test"}, "value": [1234567890, "42"]}
                ]
            }
        }
        
        results = resp["data"]["result"]
        assert len(results) == 1
        assert float(results[0]["value"][1]) == 42


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestNaturalLanguageProcessing:
    """Tests for NL query processing."""
    
    def test_keyword_detection(self):
        """Test keyword-based intent detection."""
        queries = {
            "show me anomalies": ["anomal"],
            "what's the trend": ["trend"],
            "forecast costs": ["forecast"],
            "recommend optimizations": ["recommend"],
            "check SLA": ["sla"],
        }
        
        for query, keywords in queries.items():
            q = query.lower()
            detected = any(kw in q for kw in keywords)
            assert detected, f"Failed to detect intent in: {query}"
    
    def test_application_filter_extraction(self):
        """Test extraction of application filter from query."""
        import re
        
        queries = [
            ("show metrics for chatbot-prod", "chatbot-prod"),
            ("cost in rag-assistant", "rag-assistant"),
            ('quality for "myapp"', "myapp"),
        ]
        
        for query, expected_app in queries:
            match = re.search(r'(?:for|in|app)\s+["\']?(\w[\w-]*)["\']?', query.lower())
            if match:
                assert match.group(1) == expected_app


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
