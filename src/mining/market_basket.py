"""Market Basket Analysis via FP-Growth.

Finds product pairs that are frequently bought together. Outputs
association rules sorted by lift. Degrades to a frequency-only result
when mlxtend is missing.
"""
from __future__ import annotations
import pandas as pd
try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules  # type: ignore
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False


def _basket_onehot(df, invoice_col, item_col):
    return (
        df.groupby([invoice_col, item_col])
        .size()
        .unstack(fill_value=0)
        .astype(bool)
    )


def _heuristic_pairs(df, invoice_col, item_col, top_n):
    counts = df[item_col].value_counts().head(top_n)
    return pd.DataFrame({
        "antecedents": ["(single)"] * len(counts),
        "consequents": [str(i) for i in counts.index],
        "support": counts.values / max(df[invoice_col].nunique(), 1),
        "confidence": [float("nan")] * len(counts),
        "lift": [float("nan")] * len(counts),
    })


def mine_association_rules(
    df, invoice_col="InvoiceNo", item_col="StockCode",
    min_support=0.01, min_threshold=1.0, top_n=20,
):
    """Return top-N association rules sorted by lift."""
    if not HAS_MLXTEND:
        return _heuristic_pairs(df, invoice_col, item_col, top_n)
    basket = _basket_onehot(df, invoice_col, item_col)
    basket = basket.loc[:, basket.sum() >= 3]
    if basket.shape[1] < 2:
        return _heuristic_pairs(df, invoice_col, item_col, top_n)
    freq = fpgrowth(basket, min_support=min_support, use_colnames=True)
    if len(freq) == 0:
        return _heuristic_pairs(df, invoice_col, item_col, top_n)
    rules = association_rules(freq, metric="lift", min_threshold=min_threshold)
    rules = rules.sort_values("lift", ascending=False).head(top_n)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules.reset_index(drop=True)
