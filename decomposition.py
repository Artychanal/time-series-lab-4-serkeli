from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def strength(component, residual):
    denominator = np.var(component + residual)
    return float(max(0, 1 - np.var(residual) / denominator)) if denominator > 1e-12 else 0.0


def decompose(values, dates, seasonal=13, trend=21):
    values = np.asarray(values, float)
    if len(values) < 24 or not np.isfinite(values).all():
        raise ValueError("STL requires at least two complete finite cycles")
    result = STL(values, period=12, seasonal=seasonal, trend=trend, robust=True).fit()
    frame = pd.DataFrame({"date": dates, "observed": values, "trend": result.trend,
                          "seasonal": result.seasonal, "residual": result.resid})
    frame["rolling_trend"] = pd.Series(values).rolling(13, center=True, min_periods=1).median().to_numpy()
    error = float(np.max(np.abs(values - result.trend - result.seasonal - result.resid)))
    return frame, {"trend_strength": strength(result.trend, result.resid),
                   "seasonal_strength": strength(result.seasonal, result.resid), "reconstruction_error": error,
                   "seasonal_window": seasonal, "trend_window": trend}


def steam_sensitivity(steam):
    rows = []
    for omit in (False, True):
        part = steam.iloc[int(omit):].reset_index(drop=True)
        for seasonal, trend in [(7, 19), (13, 21), (19, 25)]:
            _, info = decompose(np.log1p(part.total_reviews), part.date, seasonal, trend)
            rows.append({"omit_release": omit, "observations": len(part), **info})
    return pd.DataFrame(rows)


def sales_trend(daily):
    time = (daily.index - daily.index.min()).days.to_numpy(float)
    y = daily.revenue.to_numpy(float)
    design = np.column_stack([np.ones(len(time)), time])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    r_squared = 1 - np.sum((y - fitted)**2) / np.sum((y - y.mean())**2)
    rows = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        b = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        rows.append({"omitted_date": str(daily.index[i].date()), "slope": b[1],
                     "linear_error": float(design[i] @ b - y[i]), "constant_error": float(y[keep].mean() - y[i])})
    frame = daily.reset_index().copy()
    frame["linear_component"], frame["residual"] = fitted, y - fitted
    return frame, pd.DataFrame(rows), {"slope_per_calendar_day": float(beta[1]), "r_squared": float(r_squared),
                                       "seasonal_decomposition": "Unavailable: nine irregular dates within one month"}
