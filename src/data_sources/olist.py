"""Brazilian Olist e-commerce loader.

Olist dataset has 9 tables. For RFM-style analysis we collapse them into
the canonical transaction schema so the rest of the pipeline is
source-agnostic.

Expected files in data/raw/olist/:
    olist_orders_dataset.csv
    olist_order_items_dataset.csv
    olist_order_payments_dataset.csv
    olist_order_reviews_dataset.csv
    olist_customers_dataset.csv
    olist_products_dataset.csv
    olist_sellers_dataset.csv
    olist_geolocation_dataset.csv
    product_category_name_translation.csv

Download: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.config import PROJECT_ROOT, get_config
from .base import BaseDataSource, Dataset, register


@register
class OlistSource(BaseDataSource):
    name = "olist"

    def load(self):
        cfg = get_config().data
        root = PROJECT_ROOT / cfg.olist_raw_dir
        orders = self._read(root, "olist_orders_dataset.csv")
        items = self._read(root, "olist_order_items_dataset.csv")
        customers = self._read(root, "olist_customers_dataset.csv")
        products = self._read(root, "olist_products_dataset.csv")
        df = items.merge(
            orders[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]],
            on="order_id", how="left",
        )
        df = df.merge(
            customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id", how="left",
        )
        df = df.merge(
            products[["product_id", "product_category_name"]],
            on="product_id", how="left",
        )
        df = df.rename(columns={
            "customer_unique_id": "CustomerID",
            "order_id": "InvoiceNo",
            "order_purchase_timestamp": "InvoiceDate",
            "product_id": "StockCode",
            "product_category_name": "Description",
            "price": "UnitPrice",
            "customer_state": "Country",
        })
        df["CustomerID"] = df["CustomerID"].astype(str)
        df["Quantity"] = 1
        df["TotalPrice"] = df["UnitPrice"] * df["Quantity"]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df = df[df["order_status"] == "delivered"].copy()
        return Dataset(name=self.name, transactions=df, meta={
            "files": ["orders", "items", "customers", "products"],
            "country_focus": "Brazil",
            "n_orders": df["InvoiceNo"].nunique(),
        })

    @staticmethod
    def _read(root, name):
        path = root / name
        if not path.exists():
            raise FileNotFoundError(
                "找不到 %s。请从 Kaggle 下载 Olist 数据集并解压到 %s。" % (path, root)
            )
        return pd.read_csv(path)
