"""Multi-algorithm clustering comparison.

Runs KMeans / GaussianMixture / HDBSCAN side-by-side on scaled RFM
features, evaluates each with three metrics, picks the best by composite
score, and returns the labels + UMAP projection for visualization.

Why three algorithms? KMeans is fast but assumes spherical clusters;
GMM allows elliptical clusters and gives soft assignments; HDBSCAN
finds clusters of arbitrary shape and is robust to outliers. Comparing
them on the same data surfaces which structure is real.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.preprocessing import RobustScaler
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


@dataclass
class ClusterResult:
    algorithm: str
    labels: np.ndarray
    metrics: dict
    composite_score: float


def _safe_metric(name, fn, X, labels):
    try:
        unique = set(np.unique(labels))
        if len(unique - {-1}) < 2:
            return float("nan")
        return float(fn(X, labels))
    except Exception:
        return float("nan")


def _composite(metrics):
    sil = metrics.get("silhouette", float("nan"))
    db = metrics.get("davies_bouldin", float("nan"))
    ch = metrics.get("calinski_harabasz", float("nan"))
    if np.isnan(sil):
        return float("-inf")
    sil_n = max(0.0, (sil + 1) / 2)
    db_n = 1.0 / (1.0 + db) if not np.isnan(db) else 0.5
    ch_n = min(1.0, np.log1p(max(ch, 0)) / 10) if not np.isnan(ch) else 0.5
    return 0.5 * sil_n + 0.25 * db_n + 0.25 * ch_n


def _k_of(labels, algo):
    if algo == "hdbscan":
        uniq = set(labels.tolist()) - {-1}
        return len(uniq)
    return len(set(labels.tolist()))


def compare_clusterings(
    rfm, feature_cols=None, k_range=(3, 4, 5, 6, 7),
    primary_k=4, algorithms=None, random_state=42,
):
    """Run multiple clustering algorithms and return (results, summary)."""
    feature_cols = feature_cols or ["Recency", "Frequency", "Monetary"]
    raw = rfm[feature_cols].values.astype(float)
    X = np.sign(raw) * np.log1p(np.abs(raw))
    X_scaled = RobustScaler().fit_transform(X)
    algorithms = list(algorithms or ["kmeans", "gmm"])
    if HAS_HDBSCAN and "hdbscan" not in algorithms:
        algorithms = algorithms + ["hdbscan"]
    results = []
    for algo in algorithms:
        if algo == "hdbscan":
            if not HAS_HDBSCAN:
                continue
            cluster_size = max(50, len(X_scaled) // (primary_k * 5))
            model = hdbscan.HDBSCAN(min_cluster_size=cluster_size)
            labels = model.fit_predict(X_scaled)
            metrics = {
                "silhouette": _safe_metric("silhouette", silhouette_score, X_scaled, labels),
                "davies_bouldin": _safe_metric("davies_bouldin", davies_bouldin_score, X_scaled, labels),
                "calinski_harabasz": _safe_metric("calinski_harabasz", calinski_harabasz_score, X_scaled, labels),
            }
            score = _composite(metrics)
            results.append(ClusterResult(
                algorithm=algo, labels=labels,
                metrics=metrics, composite_score=score,
            ))
            continue
        for k in k_range:
            if algo == "kmeans":
                model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = model.fit_predict(X_scaled)
            elif algo == "gmm":
                model = GaussianMixture(
                    n_components=k, random_state=random_state,
                    covariance_type="full", n_init=5,
                )
                labels = model.fit(X_scaled).predict(X_scaled)
            else:
                continue
            metrics = {
                "silhouette": _safe_metric("silhouette", silhouette_score, X_scaled, labels),
                "davies_bouldin": _safe_metric("davies_bouldin", davies_bouldin_score, X_scaled, labels),
                "calinski_harabasz": _safe_metric("calinski_harabasz", calinski_harabasz_score, X_scaled, labels),
            }
            score = _composite(metrics)
            results.append(ClusterResult(
                algorithm=algo, labels=labels,
                metrics=metrics, composite_score=score,
            ))
    summary = pd.DataFrame([
        {
            "algorithm": r.algorithm, "k": _k_of(r.labels, r.algorithm),
            **r.metrics, "composite_score": r.composite_score,
        }
        for r in results
    ]).sort_values("composite_score", ascending=False)
    return results, summary


def umap_project(X, n_components=2, random_state=42):
    """Project to 2D for visualization. Falls back to PCA if UMAP missing."""
    if HAS_UMAP:
        n_neighbors = min(15, max(2, len(X) - 1))
        reducer = umap.UMAP(
            n_components=n_components, random_state=random_state,
            n_neighbors=n_neighbors,
        )
        return reducer.fit_transform(X)
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)
