from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def design(time, model):
    time = np.asarray(time, float)
    columns = [np.ones(len(time))]
    if model == "linear_ar":
        columns.append(time / 12)
    if model == "seasonal_ar":
        columns += [np.sin(2 * np.pi * time / 12), np.cos(2 * np.pi * time / 12)]
    return np.column_stack(columns)


def steam_values(frame):
    return np.column_stack([np.log1p(frame.total_reviews), logit(np.clip(frame.positive_share, 1e-6, 1 - 1e-6))])


@dataclass
class SteamModel:
    name: str
    beta: np.ndarray
    residual: np.ndarray
    phi: np.ndarray
    innovations: np.ndarray


def fit_steam(frame, name):
    y = steam_values(frame)
    x = design(np.arange(len(y)), name)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    phi = np.sum(residual[1:] * residual[:-1], axis=0) / np.maximum(np.sum(residual[:-1]**2, axis=0), 1e-12)
    phi = np.clip(phi, -0.95, 0.95)
    innovations = residual[1:] - residual[:-1] * phi
    innovations -= innovations.mean(axis=0)
    return SteamModel(name, beta, residual, phi, innovations)


def predict_steam(model, time, previous):
    deterministic = (design([time - 1, time], model.name) @ model.beta)
    return deterministic[1] + model.phi * (previous - deterministic[0])


def generate_steam(model, dates, rng, baseline=False):
    n = len(dates)
    residual = np.empty((n, 2))
    if baseline:
        for j in range(2):
            residual[:, j] = rng.choice(model.residual[:, j], n)
    else:
        residual[0] = model.residual[rng.integers(len(model.residual))]
        for t in range(1, n):
            residual[t] = model.phi * residual[t - 1] + model.innovations[rng.integers(len(model.innovations))]
    y = design(np.arange(n), model.name) @ model.beta + residual
    if np.max(y[:, 0]) > 30:
        raise FloatingPointError("Unrealistic numerical overflow in count generation")
    total = np.maximum(1, np.rint(np.expm1(y[:, 0]))).astype(np.int64)
    positive = np.clip(np.rint(total * expit(y[:, 1])), 0, total).astype(np.int64)
    return pd.DataFrame({"date": dates, "total_reviews": total, "positive_reviews": positive,
                         "negative_reviews": total - positive, "positive_share": positive / total})


def generate_sales(source, rng, baseline=False):
    n = len(source)
    picked = rng.integers(n, size=n)
    generated = source.iloc[picked].copy().reset_index(drop=True)
    if baseline:
        quantity = rng.choice(source.quantity.to_numpy(float), n)
        rate = rng.choice(source.shipping_rate.to_numpy(float), n)
    else:
        quantity = generated.quantity.to_numpy(float)
        rate = generated.shipping_rate.to_numpy(float)
    generated["quantity"] = np.maximum(1, np.rint(quantity * np.exp(rng.normal(0, 0.08, n)))).astype(int)
    generated["shipping_rate"] = np.maximum(0, rate * np.exp(rng.normal(0, 0.03, n)))
    generated["revenue"] = (generated.price * generated.quantity).round(2)
    generated["shipping"] = (generated.revenue * generated.shipping_rate).round(2)
    generated["shipping_rate"] = generated.shipping / generated.revenue
    generated["date"] = pd.to_datetime(rng.choice(source.date.to_numpy(), n))
    generated["shipped_date"] = generated.date + pd.to_timedelta(generated.delivery_days, unit="D")
    generated["order_id"] = np.arange(1, n + 1)
    return generated
