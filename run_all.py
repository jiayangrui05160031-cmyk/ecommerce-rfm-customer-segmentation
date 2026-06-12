"""
run_all.py
==========
一键运行整个 RFM 分析 pipeline。

适用场景：
    1. 没有 Jupyter 也能跑（命令行模式）
    2. 在 CI / 自动化测试里跑
    3. 想快速看结果、不想逐个打开 notebook

运行：
    python run_all.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rfm import build_rfm_table, rfm_score, rfm_level  # noqa: E402
from data_loader import load_raw_retail  # noqa: E402

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import matplotlib
matplotlib.use("Agg")  # 无头模式，生成 PNG 不弹窗
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
IMAGES = PROJECT_ROOT / "images"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {text}")
    print("=" * 64)


def step1_clean() -> pd.DataFrame:
    banner("STEP 1/4 - 数据清洗")
    try:
        df = load_raw_retail()
    except FileNotFoundError as e:
        print(f"[X] {e}")
        sys.exit(1)
    print(f"    原始规模: {len(df):,} 行")

    before = len(df)
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    print(f"    清洗后: {len(df):,} 行 (去除 {before - len(df):,})")

    out = DATA_PROCESSED / "cleaned_retail.pkl"
    df.to_pickle(out)
    print(f"[OK] 已保存 -> {out}")
    return df


def step2_rfm(df: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 2/4 - RFM 建模")
    rfm = build_rfm_table(df)
    rfm = rfm_score(rfm)
    rfm["Customer_Level"] = rfm_level(rfm)
    print(rfm["Customer_Level"].value_counts().to_string())

    # 分布图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"],
                               ["steelblue", "seagreen", "indianred"]):
        sns.histplot(rfm[col], bins=50, ax=ax, color=color, kde=True)
        ax.set_title(f"{col} Distribution")
    plt.tight_layout()
    fig.savefig(IMAGES / "rfm_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 分布图 -> {IMAGES / 'rfm_distribution.png'}")

    out = DATA_PROCESSED / "rfm_scored.pkl"
    rfm.to_pickle(out)
    print(f"[OK] 已保存 -> {out}")
    return rfm


def step3_cluster(rfm: pd.DataFrame) -> pd.DataFrame:
    banner("STEP 3/4 - K-Means 聚类")
    if not HAS_SKLEARN:
        print("[X] 未安装 sklearn，跳过聚类")
        print("    pip install scikit-learn")
        return rfm

    X = rfm[["Recency", "Frequency", "Monetary"]].values
    # 数据严重长尾，先 log1p 压一下；K-Means 对极值敏感，log 后用 RobustScaler 比 StandardScaler 更稳
    X_log = np.log1p(X)
    X_scaled = RobustScaler().fit_transform(X_log)

    inertias, sil_scores, k_range = [], [], range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(k_range, inertias, "bo-"); axes[0].set_title("Elbow")
    axes[0].set_xlabel("k"); axes[0].grid(alpha=0.3)
    axes[1].plot(k_range, sil_scores, "go-"); axes[1].set_title("Silhouette")
    axes[1].set_xlabel("k"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(IMAGES / "k_selection.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    best_k = k_range[int(np.argmax(sil_scores))]
    # 业务优先：silhouette 给出的 k 可能偏小；强制在 silhouette 下降不超 20% 的范围内取最大 k
    # （商业上 4~6 类最便于运营落地）
    top_score = max(sil_scores)
    candidates = [k_range[i] for i, s in enumerate(sil_scores) if s >= top_score * 0.8]
    business_k = max(candidates)
    if business_k != best_k:
        print(f"    业务优先调整: silhouette 最优 k={best_k} -> 业务 k={business_k} (silhouette={sil_scores[k_range.index(business_k)]:.3f})")
    best_k = business_k
    print(f"    最终 k = {best_k} (Silhouette = {sil_scores[k_range.index(best_k)]:.3f})")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X_scaled)
    print(rfm["Cluster"].value_counts().sort_index().to_string())

    out = DATA_PROCESSED / "rfm_clustered.pkl"
    rfm.to_pickle(out)
    print(f"[OK] 已保存 -> {out}")
    return rfm


def step4_visualize(rfm: pd.DataFrame) -> None:
    banner("STEP 4/4 - 可视化 & 营销策略")
    if "Cluster" not in rfm.columns:
        print("[!] 没有聚类结果，跳过")
        return

    # 帕累托
    seg = rfm.groupby("Cluster").agg(
        Count=("Recency", "size"),
        Revenue=("Monetary", "sum"),
    ).sort_values("Revenue", ascending=False)
    seg["Pct"] = seg["Revenue"] / seg["Revenue"].sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=seg.index.astype(str), y=seg["Pct"], ax=ax, color="steelblue", hue=seg.index.astype(str), legend=False)
    for i, v in enumerate(seg["Pct"]):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    ax.set_title("Revenue Contribution by Cluster")
    ax.set_ylabel("Revenue Share (%)")
    plt.tight_layout()
    fig.savefig(IMAGES / "revenue_pareto.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 帕累托图 -> {IMAGES / 'revenue_pareto.png'}")

    # 雷达图
    from visualization import plot_rfm_radar
    rfm_norm = rfm.copy()
    rfm_norm["R_n"] = 1 - (rfm_norm["Recency"] / rfm_norm["Recency"].max()) * 4 + 1
    rfm_norm["F_n"] = (rfm_norm["Frequency"] / rfm_norm["Frequency"].max()) * 4 + 1
    rfm_norm["M_n"] = (rfm_norm["Monetary"] / rfm_norm["Monetary"].max()) * 4 + 1
    radar = rfm_norm.groupby("Cluster")[["R_n", "F_n", "M_n"]].mean().round(2)
    radar.columns = ["R", "F", "M"]

    fig = plot_rfm_radar(radar)
    # 兼容不同 matplotlib 版本返回值
    save_path = IMAGES / "rfm_radar.png"
    try:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    except AttributeError:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"[OK] 雷达图 -> {save_path}")

    # 基于实际聚类画像命名（按 M 均值排序：M 最高的=Champions，最低=Hibernating）
    rows = []
    profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    strategy_map = {
        "Champions": ("Champions 冠军 VIP", "VIP 专属客户经理 + 新品优先购 + 推荐裂变激励", "+15% 留存, +20% 客单价"),
        "Loyal": ("Loyal Customers 忠诚用户", "会员日特权 + 积分翻倍 + 满减券", "+10% 客单价, +8% 复购"),
        "New": ("New Customers 新客", "首单 7 日内 8 折复购券 + 自动化欢迎邮件", "+30% 30日复购"),
        "Hibernating": ("Hibernating 沉睡用户", "唤醒短信/邮件 + 大额折扣 + 热门推送", "5-10% 唤醒率"),
    }
    rank = profile.sort_values("Monetary", ascending=False).index.tolist()
    name_for = {rank[0]: strategy_map["Champions"],
                rank[1]: strategy_map["Loyal"],
                rank[2]: strategy_map["New"],
                rank[3]: strategy_map["Hibernating"]}

    for cid, row in seg.iterrows():
        label, action, impact = name_for[cid]
        rows.append({
            "Cluster": int(cid), "Segment": label, "Customers": int(row["Count"]),
            "Revenue": round(row["Revenue"], 2),
            "Revenue_Share_%": round(row["Pct"], 2),
            "Strategy": action, "Expected_Impact": impact,
        })
    out = DATA_PROCESSED / "marketing_strategy.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] 策略表 -> {out}")


def main() -> None:
    banner("E-COMMERCE RFM CUSTOMER SEGMENTATION PIPELINE")
    df = step1_clean()
    rfm = step2_rfm(df)
    rfm = step3_cluster(rfm)
    step4_visualize(rfm)
    banner("ALL DONE!")
    print(f"  数据: {DATA_PROCESSED}")
    print(f"  图表: {IMAGES}")
    print(f"  下一步: 看 images/ 里的图，挑 4 张贴进 README，然后 git push")


if __name__ == "__main__":
    main()
