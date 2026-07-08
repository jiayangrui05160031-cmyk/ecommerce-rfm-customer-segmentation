# 电商 RFM 用户分群与精准营销分析

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain-agnostic](https://img.shields.io/badge/Pipeline-领域无关-orange)](#-通用化-任意-entity-event-数据)
[![Tests](https://img.shields.io/badge/Tests-22%20unit%20%2B%20smoke-2088ff)](tests/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](notebooks/)

> **RFM + 多算法聚类 + CLV + Churn 预测 + 3 个 AI Agent** 的端到端客户价值分析流水线。
> 一键产出业务报告 (`reports/business_report.html`)、Gradio Chat-with-Data 演示，
> 更重要的是 —— **同一份引擎能跑零售、捐赠、SaaS 登录、内容互动**等任意"谁-何时-做了什么-值多少"的长表。

---

## ✨ 为什么是它

这是一个**真正端到端**、**真正通用**的客户分析项目，不是又一个"RFM + K-Means 入门教程"。

| 能力层 | 实现 |
|--------|------|
| **领域无关** | `SchemaMapping` + `DomainProfile` 双契约；零售 / 捐赠 / 任意 entity-event 数据同代码 |
| **特征工程** | RFM (3 维) + 行为特征 (10 维: 客单价方差/品类宽度/活跃月份/退货率/IPI) |
| **聚类对比** | K-Means / Gaussian Mixture / HDBSCAN 三模型横评，Silhouette / Davies-Bouldin / Calinski-Harabasz 三指标 |
| **CLV 建模** | BG/NBD + Gamma-Gamma (Fader & Hardie)，业界事实标准 |
| **流失预测** | LightGBM + 行为特征，输出 `churn_prob` 与 SHAP 解释 |
| **商品关联** | FP-Growth 关联规则 (mlxtend) |
| **Cohort 留存** | 三角留存矩阵 + 队列营收曲线 |
| **时序预测** | Prophet / Holt-Winters 双路径降级 |
| **AI Agent 层** | 3 个 Agent: Segment Naming / Strategy Composer (NBA) / Chat-with-Data |
| **可视化交付** | HTML 业务报告 (Jinja2) + Gradio Chat UI |
| **工程化** | 统一 config、Mock 数据集、CI smoke test、模块化、22 个单元测试 |

### 最新工程增强

- **营销经济性排序**: Next-Best-Action 现在输出 `expected_incremental_profit` 与 `campaign_priority_score`,Top 10 推荐按增量利润和流失风险排序,更贴近真实预算投放。
- **稳健流失建模**: LightGBM 在单一标签 / 小样本场景自动降级到可解释启发式分数,避免 demo 数据或冷启动数据直接报错。
- **更干净的模型评估**: HDBSCAN 只运行一次密度聚类基线,不再在不同 k 上重复相同实验。

---

## 🚀 5 分钟启动

```bash
git clone https://github.com/<you>/ecommerce-rfm-customer-segmentation.git
cd ecommerce-rfm-customer-segmentation
pip install -r requirements.txt

# 30 秒跑完整个 pipeline (mock 数据，无 key)
python run_modern.py --source mock

# 或者跑非零售的捐赠数据 (证明通用化)
python run_modern.py --source donation --skip agents forecast
```

跑完后:
- `reports/business_report.html` ← 双击打开看完整报告
- `data/processed/` ← 中间产物 (pkl)
- 控制台输出聚类对比表 + CLV/Churn 概览 + Agent 命名结果

### 一行接入你自己的数据

```python
from src import analyze, SchemaMapping

result = analyze(
    my_df,                                # 任意长表
    mapping=SchemaMapping(                # 哪一列是谁 / 何时 / 做了什么 / 值多少
        entity_id="user_id",
        event_id="session_id",
        timestamp="login_at",
        value="watch_minutes",
    ),
    steps=["features", "cluster", "clv"],
    skip_agents_for_speed=True,           # 离线 / CI 不跑 LLM
)
print(result.rfm.head())
```

完整通用化范式见 [下面](#-通用化-任意-entity-event-数据)。

---

## 🧪 测试

```bash
# 端到端 smoke (~2 秒, mock 数据 + MockLLM)
python tests/smoke_test.py
# [smoke] PASSED in 2.04s

# 22 个单元测试 (~6 秒, 覆盖 mapping/profile/analyze/run_pipeline)
python -m pytest tests/test_unified_api.py -v
# 22 passed in 5.92s
```

| 测试 | 数量 | 覆盖 |
|---|---|---|
| `tests/smoke_test.py` | 1 端到端 | mock 数据 + MockLLM 跑完整 10 步 pipeline |
| `tests/test_unified_api.py` | 22 单元 | `SchemaMapping` / `DomainProfile` / `profile_inference` / `analyze()` / `run_pipeline()` / 捐赠域 / 旧 API 兼容 |

CI 自动跑 (`.github/workflows/smoke.yml`)。

---

## 📊 数据源

通过 `--source <name>` 或 `config.yaml` 切换。**默认走 mock (开箱即用):**

| source | 描述 | 规模 | 证明 |
|--------|------|------|------|
| `mock` | 合成数据,3 个 engineered cohort (Champions/Loyal/Hibernating) | 200 客户, ~1200 行 | 跑通完整 pipeline |
| `retail_ii` | Online Retail II (UK 2009-2011) | ~106 万行, 5942 客户 | 零售域标准 benchmark |
| `olist` | Brazilian E-Commerce (2016-2018) | ~10 万订单, 9.4 万客户 | 第二个零售域来源 |
| `donation` | 合成捐赠数据 | 150 捐赠人, ~470 行 | **非零售域证明通用化** |

新增数据源: 在 `src/data_sources/` 下加一个继承 `BaseDataSource` 的类并 `@register`,把列名映射塞进 `SchemaMapping`、领域规则塞进 `DomainProfile`。下面 [通用化](#-通用化-任意-entity-event-数据) 给出完整范式。

---

## 🌐 通用化: 任意 entity-event 数据

Pipeline 不再绑死"客户-订单"语义。任何"谁-何时-做了什么-值多少"的长表都能直接喂进来。

### 1. 显式声明 mapping

```python
from src import analyze, SchemaMapping, retail_profile

mapping = SchemaMapping(
    entity_id="user_id",          # 谁
    event_id="session_id",        # 一次行为
    timestamp="login_at",          # 何时
    value="watch_minutes",         # 值多少
)

result = analyze(
    my_df,                         # 任意长表
    mapping=mapping,
    profile="retail",              # 或自定义 DomainProfile
    steps=["features", "cluster", "clv"],
    skip_agents_for_speed=True,    # 离线/CI 不跑 LLM
)
print(result.rfm.head())
```

### 2. 让 pipeline 猜 (零配置入口)

不想写 mapping？把 DataFrame 丢进去即可,启发式会按列名 + 唯一性自动猜:

```python
from src import analyze
result = analyze(my_df)   # 自动跑 profile_inference()
```

`profile_inference()` 的判定规则:
- `timestamp`: 第一个 datetime dtype 列,否则按列名 `date/time/timestamp/_at`
- `entity_id`: 唯一率在 (0.5%, 50%) 区间内、unique count 最高的列
- `event_id`: 唯一率最高、且列名含 `id/no/uid/uuid/key` 的列
- `value`: 排除 id-like 列后,按列名优先级 `total > amount > revenue > value > price`
- `country`: 列名匹配 `country/region/state/geo`

### 3. 领域规则走 DomainProfile

`DomainProfile` 把"零售专属逻辑"从通用代码里剥出来 —— 这就是"分层抽象"的核心:

```python
from src.data_sources import DomainProfile

donation_profile = DomainProfile(
    name="donation",
    value_label="TotalDonated",
    enable_basket=False,        # 捐赠没有"购物篮"
    enable_clv=True,            # 仍然算预期未来捐赠
    is_return=None,             # 捐赠不存在"退货"
)
```

`features/rfm.py` 里 `is_return = df[invoice_col].astype(str).str.startswith("C")` 这类**零售硬编码**,已挪到 `retail_profile().is_return`。换领域零改动分析代码。

### 4. 证明: 同代码跑捐赠数据

```bash
python run_modern.py --source donation --skip agents forecast
# ================================================================
#   E-COMMERCE RFM + AI AGENT PIPELINE (source=donation)
# ================================================================
#   规模: 470 行, 150 客户
#   profile: donation
#   Cluster: kmeans (composite=0.735)
#   CLV 概览: {'mean': 87.05, 'median': 88.8, ...}
#   Churn AUC: 0.97
#   Cohort 矩阵: (36, 28)
#   ALL DONE in 2.1s
```

`market_basket` **自动跳过** (捐赠域无 basket,profile 关掉了),`return_rate` 整列填 0。同一个 `run_pipeline()`,不同 `Dataset.profile` —— 这就是分层抽象的价值。

---

## 🤖 AI Agent 层 (3 个核心)

### 1. Segment Naming Agent
把 `Cluster ID = 0` 这种机器语言翻译成"高价值低频休眠客户 (P1)"这种业务语言。LLM 只读结构化画像,输出业务标签 + 优先级 + 一句话定位。`MockLLM` 兜底。

### 2. Strategy Composer Agent (Next-Best-Action)
对每个客户输出:
- `recommended_action` (推荐营销动作)
- `channel` (email/sms/app_push)
- `expected_conversion_rate`
- `expected_revenue_per_customer`
- `expected_roi`
- `reasoning` (为什么推荐这个)

LLM 按 segment 批量调用 + 规则引擎兜底,最终产出 `nba_recommendations.csv`。

### 3. Chat-with-Data Agent
Gradio UI,可以用自然语言问:
- "客户 12345 的状态" → 调 `query_customer`
- "Champions 群体表现如何" → 调 `query_segment`
- "最近营收趋势" → 调 `query_trend`

启动: `python -m app.gradio_chat`,浏览器打开 http://localhost:7860

---

## 🏗️ 架构总览

```
                         ┌──────────────────────────────────────┐
                         │  Public API                          │
                         │    src.analyze(df, mapping, profile) │
                         └────────────┬─────────────────────────┘
                                      │
                         ┌────────────▼─────────────────────────┐
                         │  src.pipeline.run_pipeline()         │
                         │  单一编排器 (CLI / analyze() 都调)  │
                         └────────────┬─────────────────────────┘
                                      │
        ┌─────────────────┬───────────┼───────────┬──────────────────┐
        ▼                 ▼           ▼           ▼                  ▼
  features/rfm.py   models/clv.py  models/    models/           mining/
  RFM + 行为特征   BG/NBD+Gamma  cluster.py  churn.py           cohort/basket/
  (mapping/profile  -Gamma         K-Means/   LightGBM           forecast
   注入)                          GMM/HDB                     
        │                 │           │           │                  │
        └─────────────────┴───────────┴───────────┴──────────────────┘
                                      │
                         ┌────────────▼─────────────────────────┐
                         │  Dataset = name + transactions +     │
                         │             mapping + profile        │
                         └────────────┬─────────────────────────┘
                                      │
        ┌────────────────────┬────────┼────────┬─────────────────────┐
        ▼                    ▼        ▼        ▼                     ▼
  retail_ii source     olist source  mock     donation           自定义 source
  CustomerID/         customer_     Customer  DonorID/        (any entity-event
  InvoiceNo/          unique_id/    ID/       DonationID/      DataFrame +
  InvoiceDate/...     order_id/...  InvoiceNo DonationDate   SchemaMapping +
                                       /...      /Amount        DomainProfile)
```

### 数据源契约 (双层)

```python
# SchemaMapping: 哪一列是哪个角色
SchemaMapping(
    entity_id="CustomerID",     # 谁
    event_id="InvoiceNo",       # 一次行为
    timestamp="InvoiceDate",    # 何时
    value="TotalPrice",         # 值多少
    item_id="StockCode",        # 购物篮才需要
    quantity="Quantity",
    country="Country",
)

# DomainProfile: 领域规则 + 特性开关
DomainProfile(
    name="retail",
    value_label="Monetary",
    is_return=lambda df, m: df[m.event_id].astype(str).str.startswith("C"),
    enable_clv=True, enable_basket=True, enable_churn=True,
)
```

### 效率优化

`run_pipeline()` 在 normalize 后做**一次**全表 groupby (`precompute_customer_events`),把 `first_ts / last_ts / n_events / total_value` 预计算挂到 `Dataset` 上,后面 RFM / CLV / Churn / IPI 全部从这张表派生,百万行 Retail-II 上少扫 3-4 次全表。

`retail_ii.py` 用显式 `dtype` + `usecols` 加载,内存减半、加载更快。

---

## 🧩 项目结构

```
ecommerce-rfm-customer-segmentation/
├── config.yaml                  # 统一配置 (数据源/LLM/算法参数)
├── requirements.txt
├── run_all.py                   # 旧版入口 (RFM+KMeans)
├── run_modern.py                # 新版入口 (argparse + glue,调 run_pipeline)
│
├── src/
│   ├── __init__.py              # 公共入口: analyze(), profile_inference()
│   ├── config.py                # AppConfig 加载
│   ├── pipeline.py              # 统一 run_pipeline() 编排器
│   ├── data_sources/            # 多数据源 + SchemaMapping/DomainProfile 契约
│   │   ├── base.py              # Dataset / SchemaMapping / DomainProfile / profile_inference
│   │   ├── retail_ii.py         # 显式 dtype+usecols 优化
│   │   ├── olist.py
│   │   ├── mock.py
│   │   └── donation.py          # 非零售示例: 捐赠数据
│   ├── features/rfm.py          # RFM + 行为特征 (接受 mapping/profile)
│   ├── models/
│   │   ├── clustering.py        # K-Means/GMM/HDBSCAN 对比
│   │   ├── clv.py               # BG/NBD + Gamma-Gamma
│   │   └── churn.py             # LightGBM
│   ├── mining/
│   │   ├── market_basket.py     # FP-Growth
│   │   ├── cohort.py            # 留存矩阵
│   │   └── forecasting.py       # Prophet/Holt-Winters
│   ├── agents/                  # AI Agent 层
│   │   ├── llm_factory.py       # DeepSeek/OpenAI/Anthropic/Ollama/Mock
│   │   ├── base.py
│   │   ├── prompts.py
│   │   ├── segment_namer.py     # Agent 1
│   │   ├── strategy_composer.py # Agent 2 (NBA)
│   │   ├── chat_agent.py        # Agent 3
│   │   └── tools.py             # Chat agent 工具集
│   └── reports/html_report.py   # Jinja2 模板报告
│
├── app/gradio_chat.py           # Chat-with-Data UI
├── tests/
│   ├── smoke_test.py            # 端到端 30 秒 smoke (2.0s 通过)
│   └── test_unified_api.py      # 22 个单元测试 (5.9s 通过)
├── templates/                   # 报告模板 (预留)
│
├── data/processed/              # 中间产物 (pkl)
├── reports/business_report.html # 最终报告 (打开即看)
├── images/                      # 静态图 (雷达/帕累托)
│
└── .github/workflows/smoke.yml  # CI
```

---

## 🔌 LLM 接入 (Mock 默认,生产可换)

默认用 MockLLM (CI 友好)。要跑真实 LLM,把 `config.yaml` 里的 `llm.provider` 改成对应名字 + 配 API key:

| Provider | 适用 | 推荐模型 | Base URL |
|---|---|---|---|
| **minimax** (推荐) | 国内直连,中文最强 | `MiniMax-M3` | `https://api.minimaxi.com/v1` |
| DeepSeek | 性价比 | `deepseek-chat` | `https://api.deepseek.com/v1` |
| 通义千问 (DashScope) | 阿里云 | `qwen-plus` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | 需代理 | `gpt-4o-mini` | `https://api.openai.com/v1` |
| Ollama (本地) | Qwen2.5 / Llama 3 | `qwen2.5:7b` | `http://localhost:11434` |

```bash
# minimax 示例
export MINIMAX_API_KEY="<your-key>"
# config.yaml: provider: minimax, model: MiniMax-M3, base_url: https://api.minimaxi.com/v1
```

> ⚠️ **请勿把 API key 直接写进代码 / 配置文件 / README**。本仓库的 `config.yaml` 只引用环境变量名 (`api_key_env`),运行前请通过 shell 注入。

---

## 📦 主要交付物

| 文件 | 价值 |
|------|------|
| `reports/business_report.html` | **杀手锏**: 1 个 HTML = 完整业务报告 (KPI/分群/CLV/Churn/NBA/Chat 示例) |
| `data/processed/nba_recommendations.csv` | 每个客户一行营销策略 + ROI |
| `app/gradio_chat.py` | 演示页: 浏览器跟你的数据对话 |
| `tests/smoke_test.py` | 30 秒验证全 pipeline |
| `tests/test_unified_api.py` | 22 个单元测试, 覆盖通用化契约 |
| `reports/clustering_comparison.csv` | K-Means/GMM/HDBSCAN 三模型对比表 |

---

## 🗺️ 下一步路线

- [ ] 接入真实 LLM 后用 Analyst Debate Team 提升结论可信度
- [ ] 加入 Anomaly Hunter Agent (自动发现 segment 异常)
- [ ] 把 HTML 报告升级为 Plotly 交互式 dashboard
- [ ] 接入 Vector DB 做分析结果的 RAG 检索
- [ ] 在 `DomainProfile` 里加 `value_label`,驱动报告里"M"列的显示 ("Monetary" / "TotalDonated" / ...)

---

## 📄 License

MIT. 数据: UCI Online Retail II / Kaggle Brazilian E-Commerce (Olist).
