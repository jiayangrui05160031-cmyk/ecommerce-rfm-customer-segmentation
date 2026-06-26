# 电商 RFM 用户分群与精准营销分析

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](notebooks/)

> **RFM + 多算法聚类 + CLV + Churn 预测 + 3 个 AI Agent** 的端到端电商用户价值分析流水线。
> 一键产出业务报告 (`reports/business_report.html`) 和 Gradio Chat-with-Data 演示 (`python -m app.gradio_chat`)。

---

## 为什么是它

这是一个**真正端到端**的电商客户分析项目，不是又一个"RFM + K-Means 入门教程"：

| 能力层 | 实现 |
|--------|------|
| **特征工程** | RFM (3 维) + 行为特征 (10 维: 客单价方差/品类宽度/活跃月份/退货率/IPI) |
| **聚类对比** | K-Means / Gaussian Mixture / HDBSCAN 三模型横评，Silhouette / Davies-Bouldin / Calinski-Harabasz 三指标 |
| **CLV 建模** | BG/NBD + Gamma-Gamma (Fader & Hardie)，业界事实标准 |
| **流失预测** | LightGBM + 行为特征，输出 `churn_prob` 与 SHAP 解释 |
| **商品关联** | FP-Growth 关联规则 (mlxtend) |
| **Cohort 留存** | 三角留存矩阵 + 队列营收曲线 |
| **时序预测** | Prophet / Holt-Winters 双路径降级 |
| **AI Agent 层** | 3 个 Agent: Segment Naming / Strategy Composer (NBA) / Chat-with-Data |
| **可视化交付** | HTML 业务报告 (Jinja2) + Gradio Chat UI |
| **工程化** | 统一 config、Mock 数据集、CI smoke test、模块化 |

---

## 5 分钟启动

```bash
git clone https://github.com/<you>/ecommerce-rfm-customer-segmentation.git
cd ecommerce-rfm-customer-segmentation
pip install -r requirements.txt
python run_modern.py --source mock     # 30 秒跑完整个 pipeline
```

跑完后:
- `reports/business_report.html` ← 双击打开看完整报告
- `data/processed/` ← 中间产物 (pkl)
- 控制台输出聚类对比表 + CLV/Churn 概览 + Agent 命名结果

---

## AI Agent 层（3 个核心）

### 1. Segment Naming Agent
把 `Cluster ID = 0` 这种机器语言翻译成"高价值低频休眠客户 (P1)"这种业务语言。
LLM 只读结构化画像，输出业务标签 + 优先级 + 一句话定位。`MockLLM` 兜底。

### 2. Strategy Composer Agent (Next-Best-Action)
对每个客户输出:
- `recommended_action` (推荐营销动作)
- `channel` (email/sms/app_push)
- `expected_conversion_rate`
- `expected_revenue_per_customer`
- `expected_roi`
- `reasoning` (为什么推荐这个)

LLM 按 segment 批量调用 + 规则引擎兜底，最终产出 `nba_recommendations.csv`。

### 3. Chat-with-Data Agent
Gradio UI，可以用自然语言问:
- "客户 12345 的状态" → 调 `query_customer`
- "Champions 群体表现如何" → 调 `query_segment`
- "最近营收趋势" → 调 `query_trend`

启动: `python -m app.gradio_chat`，浏览器打开 http://localhost:7860

---

## Smoke Test

端到端 30 秒验证 (`tests/smoke_test.py`):

- 用 mock 数据 (200 客户) + MockLLM 跑完整 10 步 pipeline
- 不依赖任何 API key 或网络
- 验证关键产物 (HTML report / NBA csv) 生成
- CI 自动跑 (`.github/workflows/smoke.yml`)

```bash
python tests/smoke_test.py
# [smoke] PASSED in 2.09s
```

---

## 数据源

通过 `config.yaml` 切换，默认走 mock (开箱即用):

| source | 描述 | 规模 |
|--------|------|------|
| `mock` | 合成数据，含 3 个 engineered cohort (Champions/Loyal/Hibernating) | 200 客户, ~1200 行 |
| `retail_ii` | Online Retail II (UK 2009-2011)，仓库已有 5 个分块 CSV | ~106 万行, 5942 客户 |
| `olist` | Brazilian E-Commerce (2016-2018)，下载后放入 `data/raw/olist/` | ~10 万订单, 9.4 万客户 |
| `donation` | 合成捐赠数据，证明 pipeline 不绑死零售域 | 150 捐赠人, ~470 行 |

新增数据源: 在 `src/data_sources/` 下加一个继承 `BaseDataSource` 的类并 `@register`，并把列名映射塞进 `SchemaMapping`、领域规则塞进 `DomainProfile`。下面 [通用化](#通用化-任意-entity-event-数据) 这一节给出完整范式。

---

## 项目结构

```
ecommerce-rfm-customer-segmentation/
├── config.yaml                  # 统一配置（数据源/LLM/算法参数）
├── requirements.txt
├── run_all.py                   # 旧版入口（RFM+KMeans）
├── run_modern.py                # 新版入口（10 步完整 pipeline）
│
├── src/
│   ├── config.py                # AppConfig 加载
│   ├── pipeline.py              # 统一 run_pipeline() 编排器（CLI 和 analyze() 都调它）
│   ├── data_sources/            # 多数据源抽象层 + SchemaMapping/DomainProfile 契约
│   │   ├── base.py              # Dataset / SchemaMapping / DomainProfile / profile_inference
│   │   ├── retail_ii.py
│   │   ├── olist.py
│   │   ├── mock.py
│   │   └── donation.py          # 非零售示例：捐赠数据
│   ├── features/rfm.py          # RFM + 行为特征（接受 mapping/profile）
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

├── app/gradio_chat.py           # Chat-with-Data UI
├── tests/
│   ├── smoke_test.py            # 端到端 30 秒 smoke
│   └── test_unified_api.py      # 22 个单元测试：mapping/profile/analyze()/run_pipeline
├── templates/                   # 报告模板（预留）
│
├── data/processed/              # 中间产物 (pkl)
├── reports/business_report.html # 最终报告（打开即看）
├── images/                      # 静态图（雷达/帕累托）
│
└── .github/workflows/smoke.yml  # CI
```

---

## 通用化: 任意 entity-event 数据

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

### 2. 让 pipeline 猜

不想写 mapping？把 DataFrame 丢进去即可，启发式会按列名 + 唯一性自动猜:

```python
from src import analyze
result = analyze(my_df)   # 自动跑 profile_inference()
```

`profile_inference()` 的判定规则:
- `timestamp`: 第一个 datetime dtype 列，否则按列名 `date/time/timestamp/_at`
- `entity_id`: 唯一率在 (0.5%, 50%) 区间内、unique count 最高的列
- `event_id`: 唯一率最高、且列名含 `id/no/uid/uuid/key` 的列
- `value`: 排除 id-like 列后、按列名优先级 `total > amount > revenue > value > price`
- `country`: 列名匹配 `country/region/state/geo`

### 3. 领域规则走 DomainProfile

`DomainProfile` 把"零售专属逻辑"从通用代码里剥出来:

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

`features/rfm.py` 里的 `is_return = df[invoice_col].astype(str).str.startswith("C")` 这类零售硬编码，已挪到 `retail_profile().is_return`。换领域零改动分析代码。

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
```

`market_basket` 自动跳过（捐赠域无 basket），`return_rate` 整列填 0。同一个 `run_pipeline()`，不同 `Dataset.profile` —— 这就是分层抽象的价值。

---

## 切换到真实 LLM（MiniMax / DeepSeek / GPT-4o / Ollama）

默认用 MockLLM（CI 友好）。要跑真实 LLM，把 `config.yaml` 里的 `llm.provider` 改成对应名字 + 配 API key：

### MiniMax（默认推荐，已验证）

`MiniMaxChat` 用 `urllib` 直连 `https://api.minimaxi.com/v1`，**不需要 `openai` SDK**。模型固定 `MiniMax-Text-01`。

```powershell
$env:MINIMAX_API_KEY = "<你的 MINIMAX_API_KEY>"
# config.yaml
llm:
  provider: minimax
  model: MiniMax-Text-01
  api_key_env: MINIMAX_API_KEY
  base_url: https://api.minimaxi.com/v1
```

> ⚠️ **请勿把 API key 直接写进代码 / 配置文件 / README**。本仓库的 `config.yaml` 只引用环境变量名 (`api_key_env`)，运行前请通过 shell 注入。

跑验证（30 秒）：

```bash
python tests/smoke_real_llm.py
# [DONE] All 3 agents answered via real MiniMax LLM.
```

真实跑出的样本（200 客户 mock 数据集）：

| Agent | 输出 |
|-------|------|
| **Segment Naming** | 活跃复购客 [P1] / 高价值忠诚客 [P0] / 沉睡流失客 [P2] |
| **Strategy Composer** | 200 NBA 行，channel 分布 `app_push 80 + email 74 + in_app 46`，平均 ROI 49.16 |
| **Chat** | `query_customer` / `query_segment` / `query_trend` 三个意图都路由到正确工具 |

### DeepSeek / OpenAI / Anthropic

```bash
pip install langchain-openai litellm
export DEEPSEEK_API_KEY=sk-xxx
# config.yaml: provider: deepseek, model: deepseek-chat
```

### Ollama（本地 Qwen2.5 / Llama 3）

```bash
ollama pull qwen2.5:7b
# config.yaml: provider: ollama, model: qwen2.5:7b
# base_url 默认 http://localhost:11434
```

---

## 主要交付物

| 文件 | 价值 |
|------|------|
| `reports/business_report.html` | **杀手锏**：1 个 HTML = 完整业务报告 (KPI/分群/CLV/Churn/NBA/Chat 示例) |
| `data/processed/nba_recommendations.csv` | 每个客户一行营销策略 + ROI |
| `app/gradio_chat.py` | 演示页：浏览器跟你的数据对话 |
| `tests/smoke_test.py` | 30 秒验证全 pipeline |
| `reports/clustering_comparison.csv` | K-Means/GMM/HDBSCAN 三模型对比表 |

---

## 下一步路线

- [ ] 接入真实 LLM 后用 Analyst Debate Team 提升结论可信度
- [ ] 加入 Anomaly Hunter Agent（自动发现 segment 异常）
- [ ] 把 HTML 报告升级为 Plotly 交互式 dashboard
- [ ] 接入 Vector DB 做分析结果的 RAG 检索

---

## License

MIT. 数据: UCI Online Retail II / Kaggle Brazilian E-Commerce (Olist).


