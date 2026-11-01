from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler


def windows(steam, step=3, width=12):
    rows, profiles = [], []
    for start in range(0, len(steam) - width + 1, step):
        part = steam.iloc[start:start + width]
        y = np.log1p(part.total_reviews.to_numpy(float))
        rows.append({"start": str(part.date.iloc[0].date()), "end": str(part.date.iloc[-1].date()),
                     "level": y.mean(), "volatility": y.std(ddof=1),
                     "slope": np.polyfit(np.arange(width), y, 1)[0], "share": part.positive_share.mean()})
        profiles.append((y - y.mean()) / max(y.std(), 1e-10))
    return pd.DataFrame(rows), np.asarray(profiles)


def cluster_analysis(features, seed, max_k=4):
    features = np.asarray(features, float)
    if len(features) < 5 or not np.isfinite(features).all():
        raise ValueError("Insufficient finite clustering objects")
    rows, labels_by_key = [], {}
    scaled = StandardScaler().fit_transform(features)
    for method in ("kmeans", "ward"):
        for k in range(2, min(max_k, len(features) - 1) + 1):
            model = KMeans(k, random_state=seed, n_init=30) if method == "kmeans" else AgglomerativeClustering(k)
            labels = model.fit_predict(scaled)
            sizes = np.bincount(labels)
            score = silhouette_score(scaled, labels) if len(sizes) > 1 else -1.0
            rows.append({"method": method, "k": k, "silhouette": score, "minimum_cluster_size": int(sizes.min()),
                         "eligible": bool(sizes.min() >= 2)})
            labels_by_key[(method, k)] = labels
    table = pd.DataFrame(rows)
    valid = table.loc[table.eligible].sort_values("silhouette", ascending=False)
    best = (valid if len(valid) else table.sort_values("silhouette", ascending=False)).iloc[0]
    chosen = labels_by_key[(best.method, int(best.k))]
    k = int(best.k)
    def fit(x, random=seed):
        return (KMeans(k, random_state=random, n_init=30) if best.method == "kmeans" else AgglomerativeClustering(k)).fit_predict(x)
    stability = []
    for trial in range(10):
        labels = fit(scaled, seed + trial)
        stability.append({"kind": "seed", "trial": trial, "ari": adjusted_rand_score(chosen, labels)})
    robust = fit(RobustScaler().fit_transform(features))
    stability.append({"kind": "robust_scaler", "trial": 0, "ari": adjusted_rand_score(chosen, robust)})
    for omitted in range(len(features)):
        keep = np.arange(len(features)) != omitted
        labels = fit(StandardScaler().fit_transform(features[keep]))
        stability.append({"kind": "leave_one_object_out", "trial": omitted,
                          "ari": adjusted_rand_score(chosen[keep], labels)})
    return chosen, table, pd.DataFrame(stability), {"method": best.method, "k": k,
             "silhouette": float(best.silhouette), "minimum_cluster_size": int(best.minimum_cluster_size),
             "eligible": bool(best.eligible),
             "objects": len(features), "robust_scaler_ari": float(adjusted_rand_score(chosen, robust))}


def category_similarity(sales):
    table = sales.pivot_table(index="date", columns="category", values="revenue", aggfunc="sum", fill_value=0)
    profiles = table.to_numpy(float).T
    norm = np.linalg.norm(profiles, axis=1, keepdims=True)
    distance = squareform(pdist(profiles / np.maximum(norm, 1e-12), metric="euclidean"))
    return table, pd.DataFrame(distance, index=table.columns, columns=table.columns)
