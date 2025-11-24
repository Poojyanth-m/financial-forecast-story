# model_sarimax.py
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def fit_sarimax_auto(train_series, p_max=3, q_max=3, seasonal=True, m=12):
    best_aic = np.inf
    best_res = None
    best_order = None
    best_seasonal = None
    d = 1
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            for P in range(2):
                for Q in range(2):
                    try:
                        seasonal_order = (P, 1, Q, m) if seasonal else (0,0,0,0)
                        mod = SARIMAX(train_series, order=(p, d, q), seasonal_order=seasonal_order,
                                      enforce_stationarity=False, enforce_invertibility=False)
                        res = mod.fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_res = res
                            best_order = (p, d, q)
                            best_seasonal = seasonal_order
                    except Exception:
                        continue
    return best_res, best_order, best_seasonal

def forecast_sarimax(fitted_res, steps):
    pred = fitted_res.get_forecast(steps=steps)
    return pred.predicted_mean, pred.conf_int()
