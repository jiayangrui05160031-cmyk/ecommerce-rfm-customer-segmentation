"""Churn prediction with LightGBM.

Definition: Churned = no purchase in the last N days AND had at least
one historical purchase. N is configurable via config.yaml churn.definition_days.

Features: R, F, M, behavioural signals.
Output: per-customer churn_prob in [0,1], SHAP values for explainability.
Degrades to a heuristic score when LightGBM is missing.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


@dataclass
class ChurnResult:
    feature_cols: list
    auc: float
    importance: dict | None = None
    shap_values: np.ndarray | None = None
    fitted: bool = False


def _label(df, snapshot, definition_days):
    last = df.groupby("CustomerID")["InvoiceDate"].max()
    recency = (snapshot - last).dt.days
    return (recency >= definition_days).astype(int)


def _heuristic_prob(rfm):
    r = rfm["Recency"].astype(float)
    p90 = r.quantile(0.90) or 1
    return np.clip(r / p90, 0, 1).values


def fit_churn(
    df, rfm, definition_days=90, test_size=0.2, random_state=42,
    n_estimators=300, learning_rate=0.05,
):
    """Return (rfm_with_churn_prob, ChurnResult)."""
    snapshot = df["InvoiceDate"].max()
    labels = _label(df, snapshot, definition_days)
    rfm = rfm.copy()
    rfm["churn_label"] = rfm.index.map(labels).fillna(1).astype(int)
    candidate_cols = [
        "Recency", "Frequency", "Monetary",
        "avg_basket", "std_basket", "category_breadth",
        "n_active_months", "tenure_days", "return_rate", "avg_ipi_days",
    ]
    feat_cols = [c for c in candidate_cols if c in rfm.columns]
    X = rfm[feat_cols].fillna(0).astype(float).values
    y = rfm["churn_label"].values
    if not HAS_LGB:
        rfm["churn_prob"] = _heuristic_prob(rfm)
        return rfm, ChurnResult(
            feature_cols=feat_cols, auc=float("nan"),
            importance=None, shap_values=None, fitted=False,
        )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X)[:, 1]
    rfm["churn_prob"] = prob
    try:
        auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    except Exception:
        auc = float("nan")
    importance = dict(zip(feat_cols, model.feature_importances_.tolist()))
    shap_vals = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X[: min(500, len(X))])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        except Exception:
            shap_vals = None
    return rfm, ChurnResult(
        feature_cols=feat_cols, auc=auc,
        importance=importance, shap_values=shap_vals, fitted=True,
    )
