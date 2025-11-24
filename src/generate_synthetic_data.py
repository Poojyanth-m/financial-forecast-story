# generate_synthetic_data.py
import numpy as np
import pandas as pd

def generate_monthly_series(start='2015-01-01', periods=120, seed=42, freq='M'):
    np.random.seed(seed)
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    trend = np.linspace(50, 150, periods)
    seasonal = 10 * np.sin(2 * np.pi * (np.arange(periods) % 12) / 12)
    noise = np.random.normal(scale=5, size=periods)
    values = trend + seasonal + noise
    return pd.DataFrame({'date': dates, 'value': values})

if __name__ == "__main__":
    df = generate_monthly_series()
    df.to_csv("../data/synthetic_monthly.csv", index=False)
    print("Saved synthetic data to data/synthetic_monthly.csv")
