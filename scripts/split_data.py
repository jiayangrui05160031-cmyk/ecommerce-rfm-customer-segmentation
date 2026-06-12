"""
split_data.py
=============
把超过 25MB 的大数据集拆成多个小块，便于分批上传到 GitHub。

被拆分的文件：
    online_retail_II.xlsx      (43.5MB，仓库根目录)

输出：
    online_retail_II_part1.csv
    online_retail_II_part2.csv
    online_retail_II_part3.csv
    ...
    （5 个 part CSV 全部输出到仓库根目录）

每个分块 < 25MB（GitHub 单文件上传上限）。

使用方法：
    1. 把 online_retail_II.xlsx 放到仓库根目录
    2. python scripts/split_data.py
    3. 删除原 xlsx（git rm）
    4. git add online_retail_II_part*.csv && git push
"""

from pathlib import Path
import pandas as pd

# 路径
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "online_retail_II.xlsx"
DST_DIR = ROOT  # 输出到根目录

# 目标：每个分块不超过 20MB（留余量，避开 25MB 红线）
TARGET_CHUNK_MB = 20


def main() -> None:
    if not SRC.exists():
        print(f"[X] 找不到源文件: {SRC}")
        print("    请确认 online_retail_II.xlsx 在仓库根目录")
        return

    print(f"[..] 读取 {SRC.name} ...")
    # 原始数据集有两个 sheet，需要分别读再合并
    df1 = pd.read_excel(SRC, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(SRC, sheet_name="Year 2010-2011")
    df = pd.concat([df1, df2], ignore_index=True)

    # 列名统一（UCI 版带空格）
    df = df.rename(columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    })

    total_rows = len(df)
    print(f"    总行数: {total_rows:,}")
    print(f"    列: {list(df.columns)}")

    # 估算每行字节数，决定每个分块多少行
    sample = df.head(1000).to_csv(index=False).encode("utf-8")
    bytes_per_row = len(sample) / 1000
    rows_per_chunk = int(TARGET_CHUNK_MB * 1024 * 1024 / bytes_per_row)
    print(f"    平均每行 ~{bytes_per_row:.0f} 字节")
    print(f"    每块约 {rows_per_chunk:,} 行 (目标 ≤ {TARGET_CHUNK_MB}MB)")

    # 切分
    n_chunks = (total_rows + rows_per_chunk - 1) // rows_per_chunk
    print(f"    将拆成 {n_chunks} 个分块")
    for i in range(n_chunks):
        start = i * rows_per_chunk
        end = min((i + 1) * rows_per_chunk, total_rows)
        chunk = df.iloc[start:end]
        out_path = DST_DIR / f"online_retail_II_part{i+1}.csv"
        chunk.to_csv(out_path, index=False, encoding="utf-8")
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    [OK] {out_path.name}: {len(chunk):,} 行, {size_mb:.2f} MB")

    print("\n[完成] 拆分结束。可手动删除原 xlsx 后分批上传。")
    print("       src/data_loader.py 会自动拼接这些分块。")


if __name__ == "__main__":
    main()
