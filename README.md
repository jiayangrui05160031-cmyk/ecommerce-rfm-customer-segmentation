# ecommerce-rfm-customer-segmentation
# 电商 RFM 用户分群与精准营销分析

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](notebooks/)

> 基于 **RFM 模型 + K-Means 聚类** 的电商用户价值分群与营销策略推荐
> · 数据源: Online Retail II (UK 2009–2011, ~106 万条交易)
> · 输出: 4 类用户画像 + 营销动作清单

---

## 📑 目录

- [项目背景](#-项目背景)
- [核心结论](#-核心结论)
- [数据说明](#-数据说明)
- [技术栈](#-技术栈)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [分析流程](#-分析流程)
- [聚类与画像](#-聚类与画像)
- [营销策略输出](#-营销策略输出)
- [可视化预览](#-可视化预览)
- [常见问题](#-常见问题)
- [License](#-license)

---

## 🎯 项目背景

电商行业的核心命题: **把有限的营销预算投到最有价值的客户身上**。

本项目以经典的 **RFM 模型** (Recency 近度 / Frequency 频度 / Monetary 金额) 为基础，结合 **K-Means 聚类**，将客户分成 4 类价值群体，并针对每类群体给出可落地的营销动作建议。

**业务价值**:
- 识别 **高价值客户** (前 20%) 贡献了 60%+ 的营收 (二八法则)
- 针对 **沉睡用户** 设计低成本唤醒策略，提升 ROI
- 为新客设计 **首单复购券** 漏斗，提升 30 日留存

---

## 🏆 核心结论

基于聚类结果 (k=4, Silhouette ≈ 0.42) 与业务命名,4 类用户画像如下:

| 客户群 | 占比 | 营收贡献 | 关键特征 | 营销动作 | 预期效果 |
|--------|------|---------|---------|---------|---------|
| 🏆 **冠军 VIP** | ~20% | ~60% | 高 R 低 F 高 M | VIP 客户经理 + 新品优先购 | 留存 +15%, 客单 +20% |
| 💎 **忠诚用户** | ~25% | ~25% | 中低 R 高 F 中 M | 会员日特权 + 积分翻倍 | 客单 +10%, 复购 +8% |
| 🌱 **潜力新客** | ~25% | ~10% | 低 R 低 F 低 M | 首单 8 折复购券 + 欢迎邮件 | 30 日复购 +30% |
| 😴 **沉睡用户** | ~30% | ~5% | 高 R 低 F 低 M | 唤醒短信 + 大额折扣 | 唤醒率 5–10% |

> **二八法则验证**: 前 20% 客户贡献了 60%+ 营收; 沉睡用户占 30% 但仅贡献 5% 营收。

---

## 📊 数据说明

| 项 | 说明 |
|----|------|
| **数据源** | [Online Retail II (UCI / Kaggle)](https://www.kaggle.com/datasets/vijayuv/onlineretail) |
| **时间跨度** | 2009-12-01 ~ 2011-12-09 |
| **原始规模** | 1,067,371 条交易, 5,942 个客户 |
| **清洗后** | ~805,000 条 (去除退货/缺客户ID/异常单价) |
| **特征** | InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country |

### 数据切分处理

原始数据集 ~43MB,超过 GitHub 单文件 25MB 上限。
本项目已将其拆为 **5 个 < 20MB 的 CSV 分块** (`online_retail_II_part1.csv` ~ `part5.csv`)。
`src/data_loader.py` 自动识别并拼接。

---

## 🛠️ 技术栈

| 类别 | 工具 |
|------|------|
| **数据处理** | pandas, numpy |
| **机器学习** | scikit-learn (KMeans, StandardScaler, RobustScaler, silhouette_score, PCA) |
| **可视化** | matplotlib, seaborn, plotly |
| **开发** | Jupyter Notebook, Python 3.10+ |
| **数据 I/O** | openpyxl, xlrd |

详见 [`requirements.txt`](requirements.txt)

---

## 📁 项目结构

```
ecommerce-rfm-customer-segmentation/
├── README.md                    # 本文件
├── LICENSE                      # MIT 协议
├── .gitignore                   # 排除大文件(pkl/xlsx等)
├── requirements.txt             # 依赖清单
│
├── run_all.py                   # 一键运行(无 Jupyter 也能跑)
│
├── online_retail_II_part1.csv   # 数据分块 1 (~17MB)
├── online_retail_II_part2.csv   # 数据分块 2
├── online_retail_II_part3.csv   # 数据分块 3
├── online_retail_II_part4.csv   # 数据分块 4
├── online_retail_II_part5.csv   # 数据分块 5
│
├── src/                         # 核心代码
│   ├── data_loader.py           # 数据加载(自动识别分块)
│   ├── rfm.py                   # RFM 建模/打分/分级
│   └── visualization.py         # 可视化(雷达图/PCA 散点)
│
├── notebooks/                   # 分析流程
│   ├── 01_data_cleaning.ipynb
│   ├── 02_rfm_analysis.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_visualization.ipynb
│
├── scripts/
│   └── split_data.py            # 把 xlsx 拆成 5 个 CSV 的工具脚本
│
├── images/                      # 生成的图表
│   ├── rfm_distribution.png
│   ├── k_selection.png
│   ├── revenue_pareto.png
│   └── rfm_radar.png
│
├── data/
│   ├── raw/                     # 原始数据(空,数据放根目录)
│   └── processed/               # 清洗后数据(.gitignore 排除)
│
├── push_to_github.ps1           # 一键推送到 GitHub 脚本
└── GITHUB_GUIDE.md              # 详细部署指南
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/ecommerce-rfm-customer-segmentation.git
cd ecommerce-rfm-customer-segmentation
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 方式 A: 命令行一键运行 (推荐新手)

```bash
python run_all.py
```

会自动执行 **数据清洗 → RFM 建模 → 聚类 → 可视化** 全流程,结果输出到 `images/` 与 `data/processed/`。

### 4. 方式 B: Jupyter 交互分析

```bash
jupyter notebook
```

按顺序打开 `notebooks/01_data_cleaning.ipynb` → `02_rfm_analysis.ipynb` → `03_clustering.ipynb` → `04_visualization.ipynb`。

---

## 🔬 分析流程

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  01 数据清洗     │ →  │  02 RFM 建模     │ →  │  03 K-Means 聚类 │ →  │  04 可视化&策略  │
│                  │    │                  │    │                  │    │                  │
│ · 去重/去空      │    │ · R: 最近购买距今│    │ · log1p 压长尾  │    │ · 雷达图         │
│ · 去除退货单(C开头)│    │ · F: 购买频次    │    │ · RobustScaler   │    │ · 帕累托图       │
│ · 过滤异常值     │    │ · M: 累计消费    │    │ · Elbow + Silhouette │    · 散点图(PCA)    │
│ · 计算 TotalPrice│    │ · 1-5 打分       │    │ · 业务优先 k 选择│    │ · 营销策略输出   │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
```

### 关键方法论

- **RFM 打分**: R/M 用 `qcut` 按分位数切 5 段; F 用 `rank(method='first')` 后再 qcut 避免大量重复值集中
- **数据预处理**: 因 RFM 三列严重右偏长尾,先 `log1p` 压,再用 `RobustScaler` (比 StandardScaler 对极值更稳)
- **k 值选择**: 业务优先 —— 在 Silhouette 下降不超 20% 范围内取最大 k (商业上 4~6 类最便于运营落地)
- **业务命名**: 按 M 均值降序排序后,按位次映射到 Champions / Loyal / New / Hibernating

---

## 🧩 聚类与画像

```python
from src.data_loader import load_cleaned_retail
from src.rfm import build_rfm_table, rfm_score, rfm_level
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

df = load_cleaned_retail()
rfm = build_rfm_table(df)
rfm = rfm_score(rfm)

X = rfm[['Recency', 'Frequency', 'Monetary']].values
X_scaled = RobustScaler().fit_transform(np.log1p(X))

km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = km.fit_predict(X_scaled)
```

### 业务映射规则

| 排名 | 维度 | 业务标签 |
|------|------|---------|
| M 均值最高 | 高 R 低 F 高 M | 🏆 Champions 冠军 VIP |
| M 均值次高 | 中 R 中 F 中 M | 💎 Loyal Customers 忠诚用户 |
| M 均值次低 | 低 R 中 F 低 M | 🌱 New Customers 新客 |
| M 均值最低 | 高 R 低 F 低 M | 😴 Hibernating 沉睡用户 |

---

## 💼 营销策略输出

运行后会在 `data/processed/marketing_strategy.csv` 输出每类用户的画像与营销动作建议,格式:

| Cluster | Segment | Customers | Revenue | Revenue_Share_% | Strategy | Expected_Impact |
|---------|---------|-----------|---------|-----------------|----------|-----------------|
| 0 | Champions 冠军 VIP | 1,212 | 6,540,000 | 58.3% | VIP 专属客户经理 + 新品优先购 | +15% 留存, +20% 客单价 |
| 1 | Loyal Customers 忠诚用户 | 1,485 | 2,890,000 | 25.8% | 会员日特权 + 积分翻倍 | +10% 客单价, +8% 复购 |
| 2 | New Customers 新客 | 1,420 | 1,150,000 | 10.3% | 首单 7 日内 8 折复购券 | +30% 30 日复购 |
| 3 | Hibernating 沉睡用户 | 1,825 | 540,000 | 4.8% | 唤醒短信 + 大额折扣 | 5-10% 唤醒率 |

---

## 📈 可视化预览

| 图表 | 文件 | 说明 |
|------|------|------|
| RFM 三维分布 | `images/rfm_distribution.png` | Recency / Frequency / Monetary 的直方图 |
| K 值选择 | `images/k_selection.png` | Elbow + Silhouette 双图 |
| 帕累托图 | `images/revenue_pareto.png` | 各群体营收贡献 |
| 雷达图 | `images/rfm_radar.png` | 各群体 RFM 三维特征 |

> 跑完 `python run_all.py` 后会自动生成到 `images/` 目录。

---

## ❓ 常见问题

**Q1: KMeans 报 ConvergenceWarning 怎么办?**
```python
KMeans(n_clusters=4, n_init=10, max_iter=500, random_state=42)
```
调高 `n_init` 与 `max_iter`。

**Q2: 数据文件找不到?**
确认 `online_retail_II_part1.csv` ~ `part5.csv` 5 个分块都在仓库根目录。
或自行用 `scripts/split_data.py` 重新拆分。

**Q3: 如何换数据集?**
修改 `src/data_loader.py` 的列名映射,确保包含 `['CustomerID', 'InvoiceNo', 'InvoiceDate', 'TotalPrice']` 4 列即可。

**Q4: 想换 K-Means 为其他聚类?**
推荐替换为 `DBSCAN` (无需指定 k) 或 `GaussianMixture` (概率聚类)。`silhouette_score` 评估指标可继续使用。

**Q5: Excel 原始文件 (.xlsx) 怎么生成?**
```python
# 在 data/raw/ 放好 online_retail_II.xlsx 后
python scripts/split_data.py
```
会自动拆成 5 个 CSV 分块到仓库根目录。

---

## 📝 License

本项目采用 [MIT License](LICENSE)。数据来源: [Online Retail II (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II),仅用于学习与研究。

---

## ✨ 致谢

- 数据来源: UCI / Kaggle Online Retail II 数据集
- 方法论: RFM 模型 (Arthur Hughes, 1994)
- 工具: pandas, scikit-learn, seaborn 社区

---

<p align="center">
  如果这个项目对你有帮助,欢迎 ⭐ Star!<br>
  <sub>Built with ❤️ for data-driven marketing</sub>
</p>
