"""Online Retail II loader (the legacy dataset).

Reads the part-CSV chunks from data/retail_ii_parts_dir (default: project
root) and standardises column names. Returns a Dataset with the canonical
schema used downstream.
"""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd
from src.config import PROJECT_ROOT, get_config
from .base import BaseDataSource, Dataset, register

COLUMN_RENAMES = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}


@register
class RetailIISource(BaseDataSource):
    name = "retail_ii"

    def load(self):
        cfg = get_config().data
        parts_dir = PROJECT_ROOT / cfg.retail_ii_parts_dir
        files = sorted(glob.glob(str(parts_dir / "online_retail_II_part*.csv")))
        if not files:
            raise FileNotFoundError(
                "在 %s 下找不到任何 online_retail_II_part*.csv。"
                "请将 5 个分块放在仓库根目录，或运行 scripts/split_data.py 拆分。" % parts_dir
            )
        frames = []
        for f in files:
            df = pd.read_csv(f, encoding="utf-8")
            df = df.rename(columns=COLUMN_RENAMES)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        return Dataset(name=self.name, transactions=df, meta={
            "files": [Path(f).name for f in files],
            "country_focus": "United Kingdom",
        })
