# run_pipeline.py
import json
from pathlib import Path
import pandas as pd

from generate_synthetic_data import generate_monthly_series
from data_prep import load_and_prepare
from train_test_split import time_series_split
from model_sarimax import fit_sarimax_auto, forecast_sarimax
from evaluate import evaluate
from narrative import generate_narrative

DATA_DIR = Path("data")
RESULTS = Path("results")
DATA_DIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

def main():
    # 1. Generate synthetic data
    df = generate_monthly_series()
    csv_path = DATA_DIR / "synthetic_monthly.csv"
    df.to_csv(csv_path, index=False)
    print("Saved synthetic data ->", csv_path)

    # 2. Prepare
    prepared = load_and_prepare(csv_path)
    prepared.to_csv(RESULTS / "prepared_series.csv")
    series = prepared['value']

    # 3. Split
    train, val, test = time_series_split(series)
    train.to_frame(name='value').reset_index().to_csv(RESULTS / "train.csv", index=False)
    val.to_frame(name='value').reset_index().to_csv(RESULTS / "val.csv", index=False)
    test.to_frame(name='value').reset_index().to_csv(RESULTS / "test.csv", index=False)
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    # 4. Fit SARIMAX
    fitted_res, best_order, best_seasonal = fit_sarimax_auto(train)
    if fitted_res is None:
        raise RuntimeError("Failed to fit SARIMAX. Try editing grid search ranges.")
    with open(RESULTS / "model_summary.txt", "w") as f:
        f.write(f"Selected order: {best_order}, seasonal: {best_seasonal}\n\n")
        f.write(fitted_res.summary().as_text())
    print("Model fitted. Order:", best_order, "Seasonal:", best_seasonal)

    # 5. Forecast
    n_periods = max(len(test), 12)
    preds, conf_int = forecast_sarimax(fitted_res, steps=n_periods)
    last_date = series.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset('M'), periods=n_periods, freq='M')
    forecasts_df = pd.DataFrame({
        'date': forecast_index,
        'forecast': preds.values,
        'ci_lower': conf_int.iloc[:,0].values,
        'ci_upper': conf_int.iloc[:,1].values
    })
    forecasts_df.to_csv(RESULTS / "forecasts.csv", index=False)

    # 6. Evaluate (compare first len(test) preds)
    eval_len = min(len(test), n_periods)
    y_true = test.values[:eval_len]
    y_pred = preds.values[:eval_len]
    metrics = evaluate(y_true, y_pred)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 7. Narrative
    mean_forecast = float(preds.mean())
    hist_mean = float(series.mean())
    trend = "upward" if mean_forecast > hist_mean else "downward" if mean_forecast < hist_mean else "stable"
    narrative = generate_narrative(metrics, trend, model_info=f"order={best_order}, seasonal={best_seasonal}")
    (RESULTS / "narrative.txt").write_text(narrative)

    print("Pipeline complete. Results written to 'results/'")
    print("Metrics:", metrics)
    print("Narrative preview:")
    print(narrative[:400])

if __name__ == "__main__":
    main()
