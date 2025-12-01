import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred):
    """Calculate forecast accuracy metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        dict: MAE, RMSE, and MAPE metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return {"MAE": float('nan'), "RMSE": float('nan'), "MAPE": float('nan')}

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero in MAPE
    mask = y_true != 0
    if mask.sum() == 0:
        mape = float('nan')
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
