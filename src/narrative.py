# narrative.py
def generate_narrative(metrics, trend, model_info=None):
    parts = []
    parts.append("Executive summary:")
    parts.append(f"- Detected trend: {trend}")
    if model_info:
        parts.append(f"- Model: {model_info}")
    parts.append(f"- MAE: {metrics.get('MAE', 0):.2f}")
    parts.append(f"- RMSE: {metrics.get('RMSE', 0):.2f}")
    parts.append(f"- MAPE: {metrics.get('MAPE', 0):.2f}%")
    parts.append("Recommendations: Monitor forecast monthly and update model if performance degrades.")
    return "\\n".join(parts)
