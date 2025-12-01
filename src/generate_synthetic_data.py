import numpy as np
import pandas as pd

def generate_monthly_series(start='2015-01-01', periods=120, seed=42, freq='M'):
    """Generate synthetic monthly time series data.

    Args:
        start: Start date for the series
        periods: Number of periods to generate
        seed: Random seed for reproducibility
        freq: Frequency string (e.g., 'M' for monthly)

    Returns:
        DataFrame: Synthetic time series with date and value columns
    """
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
    print("Synthetic data saved to data/synthetic_monthly.csv")
