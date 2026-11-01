from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr
from statsmodels.tsa.stattools import acf, adfuller, kpss

from decomposition import decompose
from synthetic_data import fit_steam, predict_steam, steam_values


def describe(values):
    values = np.asarray(values, float)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std(ddof=1)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)), "min": float(values.min()), "max": float(values.max()),
            "skew": float(skew(values, bias=False)) if np.std(values) > 1e-10 else 0.0}


def stationarity(values, label):
    y = np.asarray(values, float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a = adfuller(y, maxlag=6, autolag="AIC")
        k = kpss(y, regression="c", nlags="auto")
    return {"series": label, "n": len(y), "adf_statistic": a[0], "adf_pvalue": a[1], "adf_lags": a[2],
            "kpss_statistic": k[0], "kpss_pvalue": k[1], "kpss_lags": k[2],
            "warnings": " | ".join(str(x.message) for x in caught)}


def forecast_comparison(steam):
    y = steam_values(steam)
    split = len(steam) - 12
    rows = []
    for model_name in ("constant_ar", "linear_ar", "seasonal_ar", "naive"):
        for t in range(36, split):
            pred = y[t - 1] if model_name == "naive" else predict_steam(fit_steam(steam.iloc[:t], model_name), t, y[t - 1])
            rows.append({"model": model_name, "phase": "validation", "index": t, "date": steam.date.iloc[t],
                         "actual_log": y[t, 0], "predicted_log": pred[0], "error": pred[0] - y[t, 0]})
        fitted = fit_steam(steam.iloc[:split], model_name) if model_name != "naive" else None
        for t in range(split, len(steam)):
            pred = y[t - 1] if fitted is None else predict_steam(fitted, t, y[t - 1])
            rows.append({"model": model_name, "phase": "test", "index": t, "date": steam.date.iloc[t],
                         "actual_log": y[t, 0], "predicted_log": pred[0], "error": pred[0] - y[t, 0]})
    predictions = pd.DataFrame(rows)
    comparison = predictions.groupby(["phase", "model"], sort=False).error.agg(
        n="size", rmse=lambda e: np.sqrt(np.mean(e**2)), mae=lambda e: np.mean(np.abs(e)), bias="mean").reset_index()
    eligible = comparison[(comparison.phase == "validation") & (comparison.model != "naive")]
    chosen = eligible.sort_values("rmse").iloc[0].model
    return predictions, comparison, chosen


def steam_properties(frame):
    y = np.log1p(frame.total_reviews.to_numpy(float))
    ac = acf(y, nlags=12, fft=False)
    _, dec = decompose(y, frame.date)
    return {"log_mean": y.mean(), "log_std": y.std(ddof=1), "log_median": np.median(y),
            "log_iqr": np.quantile(y, .75) - np.quantile(y, .25), "log_skew": skew(y, bias=False),
            "acf1": ac[1], "acf12": ac[12], "seasonal_strength": dec["seasonal_strength"],
            "trend_strength": dec["trend_strength"], "mean_positive_share": frame.positive_share.mean(),
            "corr_activity_share": spearmanr(y, frame.positive_share).statistic}


def sales_properties(frame, date_support):
    daily = frame.groupby("date").revenue.sum().reindex(date_support, fill_value=0).to_numpy(float)
    return {"price_mean": frame.price.mean(), "quantity_mean": frame.quantity.mean(),
            "quantity_std": frame.quantity.std(ddof=1), "revenue_mean": frame.revenue.mean(),
            "revenue_std": frame.revenue.std(ddof=1), "shipping_rate_mean": frame.shipping_rate.mean(),
            "corr_price_quantity": spearmanr(frame.price, frame.quantity).statistic,
            "corr_price_shipping_rate": spearmanr(frame.price, frame.shipping_rate).statistic,
            "daily_revenue_std": daily.std(ddof=1), "daily_max_revenue_share": daily.max() / max(daily.sum(), 1e-12)}


def compare_ensembles(real, records):
    individual = pd.DataFrame(records)
    rows = []
    for generator, group in individual.groupby("generator", sort=False):
        for metric, value in real.items():
            generated = group[metric].to_numpy(float)
            generated = generated[np.isfinite(generated)]
            if not len(generated):
                raise ValueError(f"No finite ensemble values: {generator}/{metric}")
            lo, hi = np.quantile(generated, [.025, .975])
            rows.append({"generator": generator, "metric": metric, "real": value, "synthetic_mean": generated.mean(),
                         "synthetic_q025": lo, "synthetic_q975": hi, "absolute_error": abs(generated.mean() - value),
                         "mean_absolute_deviation": float(np.abs(generated - value).mean()),
                         "contains_real": bool(lo <= value <= hi), "finite_replications": len(generated)})
    return individual, pd.DataFrame(rows)


def fixed_cluster_properties(features, reference, reference_labels):
    reference = np.asarray(reference, float)
    center = reference.mean(axis=0)
    scale = np.maximum(reference.std(axis=0), 1e-10)
    normalized = (reference - center) / scale
    groups = np.unique(reference_labels)
    prototypes = np.array([normalized[reference_labels == g].mean(axis=0) for g in groups])
    x = (np.asarray(features, float) - center) / scale
    distance = np.sqrt(np.sum((x[:, None, :] - prototypes[None, :, :])**2, axis=2))
    labels = np.argmin(distance, axis=1)
    result = {f"cluster_fraction_{int(g)}": float(np.mean(labels == i)) for i, g in enumerate(groups)}
    result["cluster_mean_distance"] = float(distance.min(axis=1).mean())
    return result


def effectiveness(records, real, metrics):
    table = pd.DataFrame(records)
    rows = []
    for metric in metrics:
        a = np.abs(table.loc[table.generator == "structured", metric].to_numpy() - real[metric])
        b = np.abs(table.loc[table.generator == "independent", metric].to_numpy() - real[metric])
        delta = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        rows.append({"metric": metric, "structured_mean_abs_error": a.mean(), "independent_mean_abs_error": b.mean(),
                     "error_reduction_percent": 100 * delta / max(b.mean(), 1e-12),
                     "independent_minus_structured_error": delta, "mc_ci_lower": delta - 1.96 * se,
                     "mc_ci_upper": delta + 1.96 * se,
                     "interpretation": "Approximate Monte Carlo mean-error uncertainty conditional on fitted data; not population inference"})
    return pd.DataFrame(rows)
