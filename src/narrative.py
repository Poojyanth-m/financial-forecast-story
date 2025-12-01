import pandas as pd

def generate_narrative(metrics, trend, model_info=None):
    """Generate a professional summary of forecast results."""

    if any(pd.isna(v) for v in metrics.values()):
        return ("Model failed to generate reliable predictions. "
                "Consider using more data points (minimum 50 observations) "
                "or try with synthetic data.")

    mape = metrics.get('MAPE', 0)
    trend_desc = {
        "upward": "upward trend",
        "downward": "downward trend",
        "stable": "stable trend"
    }.get(trend, "stable trend")

    accuracy_desc = (
        "excellent accuracy" if mape < 10 else
        "good accuracy" if mape < 20 else
        "moderate accuracy" if mape < 50 else
        "low accuracy"
    )

    narrative = f"""
Forecast Summary:
- Trend: {trend_desc}
- Model: {model_info or 'SARIMAX'}
- Performance: {accuracy_desc} (MAPE: {mape:.1f}%)

Recommendations:
- Monitor forecast performance monthly
- Update model with new data as available
- Consider additional features for improved accuracy
"""

    return narrative.strip()
