"""Forecast error metrics. Inputs are array-likes of equal length with no missing values."""

import numpy as np


def _arrays(y, yhat):
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    if y.shape != yhat.shape:
        raise ValueError(f"Shape mismatch: {y.shape} vs {yhat.shape}")
    if np.isnan(y).any() or np.isnan(yhat).any():
        raise ValueError("Metrics got NaN values; filter or fill before scoring")
    return y, yhat


def mae(y, yhat) -> float:
    """Mean absolute error, in bikes per hour."""
    y, yhat = _arrays(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    """Root mean squared error. Punishes big misses more than MAE."""
    y, yhat = _arrays(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y, yhat) -> float:
    """Total absolute error divided by total actual.

    Unlike MAPE it does not divide by each actual, so hours with 0 trips are fine. Returns NaN
    only when every actual is 0, where any percentage error is undefined.
    """
    y, yhat = _arrays(y, yhat)
    total = y.sum()
    return float(np.abs(y - yhat).sum() / total) if total > 0 else float("nan")


def skill(model_mae: float, baseline_mae: float) -> float:
    """1 - model MAE / baseline MAE. Positive means the model beats the baseline."""
    return 1 - model_mae / baseline_mae


def coverage(y, lower, upper) -> float:
    """Share of actuals inside [lower, upper]. For an 80% interval, about 0.80 is calibrated."""
    y, lower = _arrays(y, lower)
    _, upper = _arrays(y, upper)
    return float(np.mean((y >= lower) & (y <= upper)))
