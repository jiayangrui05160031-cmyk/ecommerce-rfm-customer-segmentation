"""Online Retail II loader (the legacy dataset).

Reads the part-CSV chunks from data/retail_ii_parts_dir (default: project
root) and standardises column names. Returns a Dataset with the canonical
retail schema + the default retail mapping/profile.
"""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd
from src.config import PROJECT_ROOT, get_config
from .base import (
    BaseDataSource, Dataset, register,
    retail_mapping, retail_profile,
)

COLUMN_RENAMES = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}

# Explicit dtypes — cuts memory ~50% and skips a 5M-row type-inference
# pass. Strings stay as object (no nullable StringArray before 2.0).
_RETAIL_DTYPES = {
    "Invoice": "string",
    "StockCode": "string",
    "Description": "string",
    "Quantity": "int32",
    "Customer ID": "string",   # Online Retail II has float IDs with NaN
    "Country": "string",
}
_RETAIL_USECOLS = [
    "Invoice", "StockCode", "Description",
    "Quantity", "InvoiceDate", "Price", "Customer ID", "Country",
]


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
            df = pd.read_csv(
                f,
                encoding="utf-8",
                usecols=_RETAIL_USECOLS,
                dtype=_RETAIL_DTYPES,
                low_memory=False,
            )
            df = df.rename(columns=COLUMN_RENAMES)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        # CustomerID is float in the source (NaN for guest checkouts);
        # keep it as float to preserve the NaN signal — features that
        # need a non-null id dropna explicitly.
        return Dataset(
            name=self.name,
            transactions=df,
            meta={
                "files": [Path(f).name for f in files],
                "country_focus": "United Kingdom",
            },
            mapping=retail_mapping(),
            profile=retail_profile(),
        )
