def time_series_split(series, train_frac=0.7, val_frac=0.15):
    """Split time series data into train, validation, and test sets.

    Args:
        series: Time series data
        train_frac: Fraction for training set
        val_frac: Fraction for validation set

    Returns:
        tuple: (train_series, val_series, test_series)
    """
    n = len(series)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]
    return train, val, test
