"""
可视化工具函数模块
封装常用的 RFM 聚类可视化逻辑。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_elbow(inertias: list, k_range: range) -> None:
    """绘制肘部法则图。"""
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_silhouette(scores: list, k_range: range) -> None:
    """绘制轮廓系数图。"""
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), scores, "go-", linewidth=2, markersize=8)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Different k")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_clusters_2d(rfm_scaled: np.ndarray, labels: np.ndarray) -> None:
    """
    用 PCA 将 3 维 RFM 数据降到 2 维后绘制聚类散点图。
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(rfm_scaled)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1],
        hue=labels, palette="tab10", s=40, alpha=0.7, edgecolor="white",
    )
    plt.title("Customer Clusters (PCA 2D projection)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_rfm_radar(cluster_profile: pd.DataFrame) -> None:
    """
    绘制各用户群体 RFM 平均得分的雷达图。
    cluster_profile: index=cluster 名, columns=['R','F','M']
    """
    categories = ["R", "F", "M"]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(9, 9))
    for idx, row in cluster_profile.iterrows():
        values = row.tolist()
        values += values[:1]
        plt.polar(angles, values, linewidth=2, label=str(idx))
        plt.fill(angles, values, alpha=0.10)

    plt.xticks(angles[:-1], categories, size=12)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey")
    plt.ylim(0, 5)
    plt.title("RFM Radar by Customer Segment", size=14, y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    return plt.gcf()
