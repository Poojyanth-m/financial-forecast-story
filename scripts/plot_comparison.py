# scripts/plot_comparison.py
"""
Plot model comparison: actual vs SARIMAX vs Naive vs Moving Average.
Saves plot to results/compare_plot.png
"""
import sys
from pathlib import Path

# add project root to sys.path so "src" imports work when running from scripts/
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.model_sarimax import fit_sarimax_auto, forecast_sarimax

RESULTS = Path("results")
PREPARED = RESULTS / "prepared_series.csv"

def naive_forecast(series, steps):
    last = series.iloc[-1]
    return np.array([last] * steps)

def moving_average_forecast(series, steps, window=3):
    mean_val = series.iloc[-window:].mean()
    return np.array([mean_val] * steps)

def main():
    if not PREPARED.exists():
        raise FileNotFoundError(f"Prepared series not found at {PREPARED}. Run the pipeline first.")

    df = pd.read_csv(PREPARED, parse_dates=["date"]).set_index("date")
    series = df["value"]

    # same split ratios used by pipeline
    n = len(series)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train = series.iloc[:train_end]
    test = series.iloc[val_end:]

    steps = len(test)

    # SARIMAX
    fitted_res, best_order, best_seasonal = fit_sarimax_auto(train)
    sarimax_pred, _ = forecast_sarimax(fitted_res, steps)
    sarimax_pred = sarimax_pred.values if hasattr(sarimax_pred, "values") else np.array(sarimax_pred)

    # Baselines
    naive_pred = naive_forecast(train, steps)
    ma_pred = moving_average_forecast(train, steps, window=3)

    # Prepare plot index (dates)
    test_index = test.index

    plt.figure(figsize=(10,6))
    plt.plot(test_index, test.values, label="Actual (test)", linewidth=2)
    plt.plot(test_index, sarimax_pred, label=f"SARIMAX ({best_order}, {best_seasonal})", linestyle='-', marker='o')
    plt.plot(test_index, naive_pred, label="Naive (last value)", linestyle='--', marker='x')
    plt.plot(test_index, ma_pred, label="Moving Avg (window=3)", linestyle=':', marker='s')

    plt.title("Model Comparison — Actual vs Predictions (Test set)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = RESULTS / "compare_plot.png"
    plt.savefig(out_path)
    print("Saved comparison plot to", out_path)

if __name__ == "__main__":
    main()
