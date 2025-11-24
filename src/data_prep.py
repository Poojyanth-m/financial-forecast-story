# data_prep.py
import pandas as pd

def load_and_prepare(path, date_col='date', value_col='value', freq='M'):
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df[[date_col, value_col]].dropna()
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col).asfreq(freq)
    df[value_col] = df[value_col].interpolate()
    return df
