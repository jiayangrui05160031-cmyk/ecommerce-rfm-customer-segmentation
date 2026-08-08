<p align="right"><a href="README.md">Chinese</a> | <strong>English</strong></p>

# E-commerce RFM Segmentation and Precision Marketing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-15%20automated-2088ff)](tests/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](notebooks/)

> An end-to-end customer-value pipeline combining **RFM, multiple clustering algorithms, CLV, churn prediction, and three AI agents**.

It produces an HTML business report, a Gradio chat-with-data demo, and a FastAPI service. The same engine works with retail, donations, SaaS events, content engagement, and other “who did what, when, and for how much” event tables.

## Why this project

| Layer | Implementation |
| --- | --- |
| Domain-agnostic ingestion | `SchemaMapping` plus `DomainProfile` contracts |
| Feature engineering | RFM plus behavioral features such as order variance, breadth, active months, returns, and IPI |
| Clustering | K-Means, Gaussian Mixture, and HDBSCAN compared with three quality metrics |
| CLV | BG/NBD and Gamma-Gamma models |
| Churn | LightGBM with behavioral features and SHAP explanations |
| Association rules | FP-Growth through mlxtend |
| Cohorts | Retention triangles and cohort revenue curves |
| Forecasting | Prophet with Holt-Winters fallback |
| AI agents | Segment Naming, Strategy Composer, and Chat-with-Data |
| Business API | Segments, profiles, budget-constrained NBA, and A/B decision endpoints |
| Delivery | Jinja2 HTML report, Gradio UI, and Swagger documentation |

Recent engineering work adds explicit data contracts, non-retail validation, centralized configuration, deterministic mock data, automated tests, resilient optional dependencies, and a CI smoke test that constructs the real Gradio interface.

## Five-minute start

```bash
git clone https://github.com/jiayangrui05160031-cmyk/ecommerce-rfm-customer-segmentation.git
cd ecommerce-rfm-customer-segmentation
pip install -r requirements.txt

# Complete pipeline with generated data; no key required
python run_modern.py --source mock

# Prove domain independence with donation events
python run_modern.py --source donation --skip agents forecast
```

Default outputs are written under `reports/`, including `business_report.html` and machine-readable intermediate results.

## API and chat UI

Start the service:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

Analyze an experiment:

```bash
curl -X POST http://127.0.0.1:8000/experiments/analyze \
  -H "Content-Type: application/json" \
  -d '{"control_conversions":120,"control_total":2000,"treatment_conversions":145,"treatment_total":2000}'
```

Start the Gradio chat-with-data interface:

```bash
python app/gradio_chat.py
```

The entry point resolves the repository root itself, so it can also be started from another working directory.

## Bring your own data

The input contract is a long event table. At minimum, identify:

- entity/customer identifier;
- event or transaction timestamp;
- monetary value, when applicable;
- optional transaction identifier, quantity, item, return flag, and channel fields.

### Explicit mapping

```python
from src.data.schema import SchemaMapping

mapping = SchemaMapping(
    entity_id="customer_id",
    event_time="created_at",
    value="revenue",
    event_id="order_id",
)
```

### Automatic inference

Point the pipeline at a CSV and allow the ingestion layer to infer common column roles. Review the inferred contract before using production data.

### Domain profiles

`DomainProfile` separates domain rules from physical column names. Profiles control behaviors such as negative-value handling, CLV eligibility, return logic, and feature availability without forking the pipeline.

Built-in examples demonstrate both retail transactions and donation events with the same processing code.

## AI agent layer

### Segment Naming Agent

Turns cluster statistics into stable, business-readable segment names and descriptions.

### Strategy Composer Agent

Generates next-best actions, channels, messages, and offers under budget and eligibility constraints.

### Chat-with-Data Agent

Answers natural-language questions using computed customer, segment, CLV, churn, and campaign artifacts rather than unrestricted free-form generation.

The default configuration supports a deterministic mock LLM, so the full pipeline and tests run without external credentials. Production providers are configured through environment variables or `config.yaml`; never commit API keys.

## Architecture

```text
CSV / generated events
        │
        ▼
SchemaMapping + DomainProfile
        │
        ▼
cleaning → RFM/behavior features → clustering
        │                         │
        ├── CLV / churn / cohort / forecast
        ├── association rules
        └── AI agents
                 │
                 ▼
HTML report + FastAPI + Gradio + experiment decisions
```

## Project layout

| Path | Purpose |
| --- | --- |
| `src/data/` | Loading, schema contracts, cleaning, mock data, and domain profiles |
| `src/features/` | RFM and behavioral feature engineering |
| `src/models/` | Clustering, CLV, churn, association, cohort, and forecasting |
| `src/agents/` | Segment naming, strategy composition, and chat-with-data |
| `src/reporting/` | Business-report generation and charts |
| `app/api.py` | FastAPI service |
| `app/gradio_chat.py` | Gradio interface |
| `tests/` | Pipeline, API, smoke, and experiment tests |
| `notebooks/` | Exploratory analysis |
| `reports/` | Generated deliverables |

## Testing

```bash
# Fast end-to-end smoke test with mock data and MockLLM
python tests/smoke_test.py

# Complete automated suite
python -m pytest -q
```

CI installs a minimal supported dependency set, executes the smoke test and automated tests, and constructs the real Gradio interface to catch upstream UI API changes.

## Data sources

The repository includes generated/mock inputs and scripts compatible with the Online Retail II-style schema. Large or licensed source datasets should be obtained from their original provider and kept outside Git history when redistribution is restricted.

## Main outputs

- customer-level RFM, CLV, churn, and segment tables;
- cluster comparison and model-quality metrics;
- cohort, association, forecast, and feature artifacts;
- next-best-action recommendations;
- `reports/business_report.html`;
- REST endpoints and interactive API documentation;
- Gradio chat-with-data demonstration.

## Roadmap

- Stronger data-quality diagnostics and schema review.
- Production feature-store and model-registry adapters.
- Distributed execution for larger event tables.
- Campaign feedback loops and uplift modeling.
- Additional domain profiles and multilingual reporting.

## License

Released under the [MIT License](LICENSE).
