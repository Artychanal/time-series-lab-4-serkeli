from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def correlation_tables(frame, prefix, output):
    selected = frame.select_dtypes(include="number")
    selected = selected.loc[:, selected.nunique() > 1]
    result = {}
    for method in ("pearson", "spearman"):
        result[method] = selected.corr(method=method)
        result[method].to_csv(output / f"{prefix}_{method}.csv")
    return result


def leave_one_out_correlations(frame, pairs):
    rows = []
    for left, right in pairs:
        x, y = frame[left].to_numpy(float), frame[right].to_numpy(float)
        full = float(spearmanr(x, y).statistic)
        reduced = [float(spearmanr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(len(frame))]
        rows.append({"left": left, "right": right, "n": len(frame), "spearman": full,
                     "loo_min": min(reduced), "loo_max": max(reduced)})
    return pd.DataFrame(rows)


def lagged_associations(frame, max_lag=3):
    x = np.log1p(frame.total_reviews).diff()
    y = frame.positive_share.diff()
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        pair = pd.concat([x.rename("x"), y.shift(-lag).rename("y")], axis=1).dropna()
        rows.append({"lag_months": lag, "meaning": "corr(delta log total[t], delta share[t+lag])",
                     "n": len(pair), "pearson": pair.x.corr(pair.y), "spearman": pair.x.corr(pair.y, method="spearman")})
    return pd.DataFrame(rows)
