from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def fit_sarimax_auto(train_series, p_max=3, q_max=3, seasonal=True, m=12):
    """Automatically fit SARIMAX model with grid search.

    Args:
        train_series: Training time series data
        p_max: Maximum AR order
        q_max: Maximum MA order
        seasonal: Whether to include seasonal components
        m: Seasonal period

    Returns:
        tuple: (fitted_model, best_order, best_seasonal_order)
    """
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
                        seasonal_order = (P, 1, Q, m) if seasonal else (0, 0, 0, 0)
                        model = SARIMAX(train_series, order=(p, d, q),
                                       seasonal_order=seasonal_order,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
                        result = model.fit(disp=False)
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_res = result
                            best_order = (p, d, q)
                            best_seasonal = seasonal_order
                    except:
                        continue

    return best_res, best_order, best_seasonal

def forecast_sarimax(fitted_res, steps):
    """Generate forecasts from fitted SARIMAX model.

    Args:
        fitted_res: Fitted SARIMAX model
        steps: Number of steps to forecast

    Returns:
        tuple: (predictions, confidence_intervals)
    """
    pred = fitted_res.get_forecast(steps=steps)
    return pred.predicted_mean, pred.conf_int()
