"""
RFM 工具函数模块
封装常用的 RFM 计算与打分逻辑，便于在 notebook 中复用。
"""

import pandas as pd
import numpy as np


def build_rfm_table(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    从清洗后的交易数据构建 RFM 基础表。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'TotalPrice'] 四列
    snapshot_date : pd.Timestamp, optional
        分析基准日，默认取数据中最大日期 + 1 天

    Returns
    -------
    pd.DataFrame
        以 CustomerID 为索引，包含 Recency / Frequency / Monetary 三列
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    return rfm


def rfm_score(rfm: pd.DataFrame, labels=None) -> pd.DataFrame:
    """
    对 RFM 三个维度按分位数打分（1-5 分），并生成 RFM_Score 字符串。
    R/M 用 qcut 按升序切 5 段（数值越大分越高）；
    F 用 rank(method='first') 后再 qcut 避免大量重复值落入同一区间。
    """
    labels = labels or [1, 2, 3, 4, 5]

    rfm = rfm.copy()
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=labels)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), 5, labels=labels
    )
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=labels)

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    return rfm


def rfm_level(rfm: pd.DataFrame) -> pd.Series:
    """
    根据 RFM 综合分（三个维度之和）划分客户等级。
    13-15 分为冠军客户，10-12 分为高价值，7-9 分为中等，<=6 分为低价值。
    """
    total = rfm["R_Score"].astype(int) + rfm["F_Score"].astype(int) + rfm["M_Score"].astype(int)
    return pd.cut(
        total,
        bins=[0, 6, 9, 12, 15],
        labels=["Low", "Mid", "High", "Champion"],
    ).rename("Customer_Level")
