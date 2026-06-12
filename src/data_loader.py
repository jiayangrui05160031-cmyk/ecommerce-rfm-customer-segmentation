"""
data_loader.py
==============
统一的数据加载入口。

设计动机：
    Online Retail II 数据集（~1,067,371 行）超过 GitHub 单文件 25MB 上限。
    我们把它拆成多个 < 20MB 的 CSV 分块，直接放在仓库根目录。
    加载时这个模块会自动：
        1. 优先尝试加载 online_retail_II_part*.csv（仓库根目录的分块）
        2. 如果没有，提示用户先下载数据

使用：
    from data_loader import load_raw_retail
    df = load_raw_retail()  # 返回完整 DataFrame
"""

from pathlib import Path
import glob
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PART_GLOB = PROJECT_ROOT / "online_retail_II_part*.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _find_part_files() -> list[Path]:
    """
    找到仓库根目录下所有 online_retail_II_part*.csv，按文件名排序。
    """
    files = sorted(glob.glob(str(DATA_PART_GLOB)))
    return [Path(f) for f in files]


def load_raw_retail() -> pd.DataFrame:
    """
    加载原始交易数据。从仓库根目录的 part CSV 拼接读取。

    Returns
    -------
    pd.DataFrame
        包含所有原始交易记录

    Raises
    ------
    FileNotFoundError
        找不到任何数据文件时
    """
    parts = _find_part_files()
    if not parts:
        raise FileNotFoundError(
            f"在 {PROJECT_ROOT} 下找不到任何 online_retail_II_part*.csv 文件。\n"
            f"  请确认 5 个 part CSV 在仓库根目录，"
            f"或从 https://www.kaggle.com/datasets/vijayuv/onlineretail 下载后用 scripts/split_data.py 拆分"
        )
    print(f"[data_loader] 拼接 {len(parts)} 个分块 CSV ...")
    frames = []
    for p in parts:
        print(f"  - {p.name}")
        df_part = pd.read_csv(p, encoding="utf-8")
        # CSV 不保留 datetime 类型，强制转
        df_part["InvoiceDate"] = pd.to_datetime(df_part["InvoiceDate"])
        frames.append(df_part)
    df = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] 拼接完成: {len(df):,} 行")
    return df


def load_cleaned_retail() -> pd.DataFrame:
    """
    加载清洗后的数据。优先读 pkl 缓存，否则从原始数据重新跑清洗。

    这个回退机制让仓库不需要保存 51MB 的 pkl：
    用户 clone 下来后，第一次跑会自动重建；之后直接读 pkl 缓存。
    """
    pkl_path = DATA_PROCESSED / "cleaned_retail.pkl"
    if pkl_path.exists():
        print(f"[data_loader] 读取已清洗缓存: {pkl_path.name}")
        return pd.read_pickle(pkl_path)

    print(f"[data_loader] 未找到 {pkl_path.name}，从原始数据重新清洗 ...")
    raw = load_raw_retail()

    before = len(raw)
    raw = raw.dropna(subset=["CustomerID"])
    raw = raw[~raw["InvoiceNo"].astype(str).str.startswith("C")]
    raw = raw[(raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)]
    raw["CustomerID"] = raw["CustomerID"].astype(int)
    raw["TotalPrice"] = raw["Quantity"] * raw["UnitPrice"]
    print(f"[data_loader] 清洗完成: {len(raw):,} 行 (去除 {before - len(raw):,})")

    # 自动缓存，下次直接读
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(pkl_path)
    print(f"[data_loader] 已缓存 -> {pkl_path.name}")
    return raw


if __name__ == "__main__":
    # 单独运行此脚本可测试加载器
    print("测试 load_raw_retail() ...")
    df = load_raw_retail()
    print(f"形状: {df.shape}")
    print(f"列: {list(df.columns)}")
    print(df.head())
