"""
verify_pipeline.py
==================
无数据集也能跑：用 mock 数据验证整个 RFM + 聚类流程是否正确。

用途：
    1. 新手 clone 项目后第一件事（不用下载 50 万行数据）
    2. CI 自动化测试（每次 push 都跑一遍）
    3. 验证 src/ 模块和 pipeline 逻辑

运行：
    python verify_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 把 src/ 加入路径
sys.path.insert(0, str(Path(__file__).parent))

from rfm import build_rfm_table, rfm_score, rfm_level

# sklearn 是可选的；没装就跳过聚类步骤
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def make_mock_retail(n_customers: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    模拟 Online Retail II 风格的数据：
    - 每个客户有自己的"活跃度"和"客单价"参数
    - 据此生成随机订单
    """
    rng = np.random.default_rng(seed)
    customer_ids = rng.integers(10000, 99999, size=n_customers)

    # 每个客户分到一个 segment 参数
    activity = rng.choice(["high", "mid", "low"], size=n_customers, p=[0.2, 0.5, 0.3])
    aov = rng.lognormal(mean=4.0, sigma=0.8, size=n_customers)  # 平均客单价 ~55 GBP

    rows = []
    base_date = pd.Timestamp("2011-11-01")
    for cid, act, price in zip(customer_ids, activity, aov):
        n_orders = {"high": rng.integers(5, 20),
                    "mid": rng.integers(2, 8),
                    "low": rng.integers(0, 3)}[act]
        for _ in range(n_orders):
            days_ago = int(rng.integers(0, 365))
            qty = int(rng.integers(1, 20))
            unit_price = max(0.5, price / qty * rng.uniform(0.8, 1.2))
            rows.append({
                "InvoiceNo": f"INV{rng.integers(100000, 999999)}",
                "StockCode": f"SKU{rng.integers(1000, 9999)}",
                "Description": "Mock Product",
                "Quantity": qty,
                "InvoiceDate": base_date - pd.Timedelta(days=days_ago),
                "UnitPrice": round(unit_price, 2),
                "CustomerID": int(cid),
                "Country": rng.choice(["United Kingdom", "Germany", "France"]),
            })

    df = pd.DataFrame(rows)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # 故意混入一些缺失和退货，便于测试清洗逻辑
    sample_idx = rng.choice(len(df), size=10, replace=False)
    df.loc[sample_idx[:5], "CustomerID"] = np.nan
    sample_idx2 = rng.choice(len(df), size=5, replace=False)
    for i in sample_idx2:
        df.loc[i, "InvoiceNo"] = "C" + str(df.loc[i, "InvoiceNo"])
        df.loc[i, "Quantity"] = -df.loc[i, "Quantity"]

    return df


def verify() -> None:
    print("=" * 60)
    print("  RFM Pipeline 验证脚本 (mock data)")
    print("=" * 60)

    # 1. 构造 mock 数据
    print("\n[1/5] 生成 mock 数据 ...")
    raw = make_mock_retail()
    print(f"      原始规模: {len(raw)} 行")
    assert "CustomerID" in raw.columns, "缺 CustomerID 列"
    assert "TotalPrice" in raw.columns, "缺 TotalPrice 列"
    print("      [OK] schema 正确")

    # 2. 清洗
    print("\n[2/5] 模拟清洗（去退货 + 缺失值）...")
    cleaned = raw.dropna(subset=["CustomerID"])
    cleaned = cleaned[~cleaned["InvoiceNo"].astype(str).str.startswith("C")]
    cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)]
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(int)
    print(f"      清洗后: {len(cleaned)} 行")
    assert len(cleaned) > 0, "清洗后没有数据，pipeline 出错"
    print("      [OK] 清洗通过")

    # 3. RFM
    print("\n[3/5] 计算 RFM ...")
    rfm = build_rfm_table(cleaned)
    print(f"      客户数: {len(rfm)}")
    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
    assert (rfm["Monetary"] > 0).all(), "Monetary 应为正"
    print("      [OK] RFM 计算正确")

    # 4. 打分
    print("\n[4/5] RFM 打分 ...")
    rfm = rfm_score(rfm)
    rfm["Customer_Level"] = rfm_level(rfm)
    assert rfm["RFM_Score"].notna().all(), "RFM_Score 有空值"
    print(f"      等级分布:\n{rfm['Customer_Level'].value_counts().to_string()}")
    print("      [OK] 打分通过")

    # 5. 聚类（可选）
    print("\n[5/5] K-Means 聚类 ...")
    if not HAS_SKLEARN:
        print("      未安装 sklearn，跳过聚类步骤（pip install scikit-learn 可启用）")
        print("      [OK] 基础 RFM 流程已通过")
    else:
        X = rfm[["Recency", "Frequency", "Monetary"]].values
        X_scaled = StandardScaler().fit_transform(X)

        best_k, best_score = 3, -1
        for k in range(2, 8):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            s = silhouette_score(X_scaled, labels)
            if s > best_score:
                best_k, best_score = k, s

        print(f"      最优 k={best_k} (Silhouette={best_score:.3f})")
        assert best_score > 0, "聚类质量太差"
        print("      [OK] 聚类通过")

    print("\n" + "=" * 60)
    print("  [OK] 全部通过！项目代码逻辑可正常运行")
    print("  下一步：下载真实数据替换 mock 数据")
    print("=" * 60)


if __name__ == "__main__":
    verify()
