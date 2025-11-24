import json
from pathlib import Path
import pandas as pd
import numpy as np

from model_sarimax import fit_sarimax_auto, forecast_sarimax
from evaluate import evaluate


RESULTS = Path("results")
DATA = RESULTS / "prepared_series.csv"

def naive_forecast(series, steps):
    """Naive forecast: always predicts last value."""
    last = series.iloc[-1]
    return np.array([last] * steps)

def moving_average_forecast(series, steps, window=3):
    """Moving average of last N points."""
    mean_val = series.iloc[-window:].mean()
    return np.array([mean_val] * steps)

def main():
    # Load prepared data
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    series = df["value"]

    # Train-test split (same ratios used before)
    n = len(series)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train = series.iloc[:train_end]
    test = series.iloc[val_end:]

    # 1 — SARIMAX model
    sarimax_res, best_order, best_seasonal = fit_sarimax_auto(train)
    steps = len(test)
    preds_sarimax, _ = forecast_sarimax(sarimax_res, steps)
    sarimax_metrics = evaluate(test.values, preds_sarimax.values)

    # 2 — Naive baseline
    preds_naive = naive_forecast(train, steps)
    naive_metrics = evaluate(test.values, preds_naive)

    # 3 — Moving average baseline
    preds_ma = moving_average_forecast(train, steps, window=3)
    ma_metrics = evaluate(test.values, preds_ma)

    comparison = {
        "sarimax": sarimax_metrics,
        "naive": naive_metrics,
        "moving_average": ma_metrics,
        "selected_order": str(best_order),
        "selected_seasonal": str(best_seasonal)
    }

    with open(RESULTS / "compare_metrics.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nModel Comparison Completed!")
    print("\nSARIMAX:", sarimax_metrics)
    print("Naive:", naive_metrics)
    print("Moving Average:", ma_metrics)
    print("\nResults saved to results/compare_metrics.json")

if __name__ == "__main__":
    main()
