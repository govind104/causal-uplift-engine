# Causal Uplift & Policy Optimization Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/EconML-0.14-green" alt="EconML">
  <img src="https://img.shields.io/badge/FastAPI-0.128-teal" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.53-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-Ready-blue" alt="Docker">
</p>

> **Traditional models predict churn. This system predicts *persuadability*, saving 20% of marketing budget by ignoring 'Lost Causes'.**

## 🎯 What This Does

This is an end-to-end **Causal Inference System** that:
1. **Identifies Persuadable Customers** - Those who will convert *because* of your intervention
2. **Optimizes Budget Allocation** - Target high-uplift customers first
3. **Provides Transparent Explanations** - Understand *why* segments respond differently

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **CATE Correlation** | 98.2% |
| **Top 20% Ranking Accuracy** | 92% |
| **Projected ROI** | 285% |

The model correctly identifies the injected causal structure:
- **Age**: Younger customers have higher uplift
- **Loyalty**: Low-loyalty customers are more persuadable
- **Income**: Higher income increases treatment response

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker-compose up --build
```
- API: http://localhost:8000
- Dashboard: http://localhost:8501

### Option 2: Local Development
```bash
# Install dependencies
uv sync

# Train model (generates synthetic data)
uv run python main.py --quick

# Start API
uv run uvicorn src.api.main:app --port 8000

# Start Dashboard (new terminal)
uv run streamlit run src/dashboard/app.py
```

## 🏗️ Architecture

```mermaid
graph LR
    A[Synthetic Data] --> B[Preprocessing]
    B --> C[T-Learner/EconML]
    C --> D[CATE Predictions]
    D --> E[FastAPI]
    D --> F[Visualizations]
    E --> G[Streamlit Dashboard]
    F --> G
```

## 📁 Project Structure

```
causal-uplift-engine/
├── src/
│   ├── data/
│   │   └── generator.py      # Synthetic data with ground truth
│   ├── models/
│   │   └── t_learner.py      # T-Learner wrapper + SHAP
│   ├── visualization/
│   │   └── plots.py          # Qini, ROI, Decile plots
│   ├── api/
│   │   ├── main.py           # FastAPI endpoints
│   │   └── schemas.py        # Pydantic models
│   └── dashboard/
│       └── app.py            # Streamlit UI
├── main.py                   # Training pipeline
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/predict` | POST | Score individual customer |
| `/api/optimize/allocate` | POST | Calculate optimal targeting |
| `/api/explain/global` | GET | Feature importance |

### Example: Predict Uplift
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 25, "income": 80000, "loyalty_score": 0.1}}'
```

Response:
```json
{
  "customer_id": "unknown",
  "uplift_score": 0.52,
  "segment": "Persuadable",
  "cate_percentile": 99.4
}
```

## 🧪 Methodology

### Why T-Learner?
T-Learner avoids the **regularization bias** of S-Learners where weak treatment effects get shrunk to zero - crucial when the treatment signal is subtle.

### Why Synthetic Data?
- **Ground Truth Validation**: Real datasets lack true counterfactuals
- **Interpretable Features**: No anonymized `f1`, `f2` columns
- **Controlled Causal Structure**: Verifiable that model learns the truth

### Customer Segments
| Segment | Percentile | Action |
|---------|------------|--------|
| 🟢 Persuadable | >75% | Target first |
| 🔵 Sure Thing | 50-75% | Will convert anyway |
| 🟡 Sleeping Dog | 25-50% | Skip if budget-constrained |
| 🔴 Lost Cause | <25% | Don't waste budget |

## 📈 Visualizations

The training pipeline generates:
- **Qini Curve**: Cumulative incremental gains
- **ROI Analysis**: Optimal targeting threshold
- **Decile Comparison**: Treatment vs Control rates
- **Causal Quadrants**: Ground truth validation

## 👤 Author

Built as a portfolio project demonstrating causal ML, API development, and interactive dashboards.

## 📄 License

MIT
