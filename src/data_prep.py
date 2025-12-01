import pandas as pd

def load_and_prepare(data, date_col='date', value_col='value', freq='M'):
    """Load and prepare time series data for forecasting.

    Args:
        data: DataFrame or file path to CSV
        date_col: Name of date column
        value_col: Name of value column
        freq: Frequency for resampling (e.g., 'M' for monthly)

    Returns:
        DataFrame: Prepared time series with datetime index
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data, parse_dates=[date_col])

    df = df[[date_col, value_col]].dropna()

    # Handle duplicate dates by aggregating
    if df[date_col].duplicated().any():
        df = df.groupby(date_col)[value_col].sum().reset_index()

    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col).asfreq(freq)
    df[value_col] = df[value_col].interpolate()

    return df
