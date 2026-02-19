# 💳 Credit Card Fraud Detection — End-to-End ML Engineering System

A production-oriented machine learning system demonstrating:

- Multi-model training & evaluation  
- Automated model selection & model registry  
- ZenML pipeline orchestration  
- MLflow experiment tracking & registry  
- DVC for data versioning (optional remote)  
- Dockerized inference API (FastAPI)  
- CI/CD with GitHub Actions  
- Public interactive Streamlit dashboard (Streamlit Community Cloud)

---

## 🔗 Live Demo

**Interactive Dashboard:** https://credit-fraud-ml-dashboard.streamlit.app/

The dashboard includes:
- Model performance comparison  
- Feature importance visualization  
- Live fraud risk prediction with risk interpretation

---

## 🏗 Architecture Overview

Data (tracked with DVC)
          ↓
ZenML Pipeline (train & evaluate)
          ↓
Multi-Model Training (LR, KNN, DT, RF)
          ↓
MLflow Tracking → Model Registry
          ↓
Automated Best Model Selection → exported_model/
          ↓
- Streamlit Dashboard (Public Demo)
- FastAPI Inference Service (Dockerized)
- **CI/CD:** GitHub Actions (lint/build/smoke-tests)


---

## 📊 Models Trained

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest (Production)

### ✅ Production Model: Random Forest

Selected based on weighted evaluation metrics.

**Performance (Synthetic Dataset)**  
- AUC: 1.00  
- Precision: 1.00  
- Recall: 0.9999  
- F1 Score: 0.9999

> ⚠ **Note:** The dataset used for this project is synthetically generated for demonstration purposes. Near-perfect metrics reflect synthetic data characteristics and should **not** be interpreted as real-world fraud detection performance.

---

## 🔍 Live Fraud Risk Prediction (Streamlit)

Interactive inputs with short hover-help for each feature:

- `distance_from_home` — distance between transaction and registered home  
- `distance_from_last_transaction` — distance from previous transaction  
- `ratio_to_median_purchase_price` — transaction amount / median purchase price  
- `repeat_retailer` — 1 if previously transacted with this retailer, else 0  
- `used_chip` — 1 if chip used, else 0  
- `used_pin_number` — 1 if PIN entered, else 0  
- `online_order` — 1 if online, else 0

Outputs:
- Fraud probability (0–100%)  
- Risk level interpretation: Low / Moderate / High

---

## ⚙️ Tech Stack & Skills Demonstrated

**Core Tools**
- Python, Pandas, NumPy, scikit-learn  
- MLflow (experiments & registry)  
- ZenML (pipeline orchestration)  
- DVC (data versioning & reproducibility)  
- Docker (containerization)  
- FastAPI (production inference)  
- Streamlit (public interactive demo)  
- Git & GitHub (code/version control)  
- GitHub Actions (CI/CD pipelines)

**Additional / Ops**
- MLflow backend options (filesystem, sqlite, or remote DB)  
- Artifact stores: local exported_model/ (committed) or S3/GCS for production  
- Basic observability: logging, model metrics, and smoke tests  
- Packaging & environment: requirements files, minimal production deps

**Skills**
- Reproducible ML pipelines, experiment tracking, model selection  
- Model export & containerized serving, API design for inference  
- CI for reproducible builds and deployment validation  
- UX for client-facing demos and interpretability (feature importance, short explanations)  
- Data versioning and handling synthetic vs. real data caveats

---

## 🚀 Quick Start (run locally)

> Recommended: use a virtual environment

```bash
# 1. setup
python -m venv Credit
source Credit/bin/activate
pip install -r requirements.txt

# 2. run training pipeline (ZenML pipeline that logs to MLflow)
python pipeline.py

# 3. run API (serves exported_model/)
uvicorn app.model_server:app --reload --port 8000

# 4. run Streamlit demo (loads exported_model/)
streamlit run streamlit_app.py
'''

🧩 Repo Contents 

- app/ — FastAPI model server

- exported_model/ — production-ready model artifact (committed)

- streamlit_app.py — public UI for demonstration

- pipeline.py, models.py, EDA.py, data_ingestion.py — training & pipeline code

- .github/workflows/ci.yml — CI pipeline (build + lint + docker)

- requirements.txt (demo / api deps)

- README.md

-----------


🔒 Best Practices & Notes

- Do not commit runtime caches: mlruns/, .zen/, .dvc/cache/ should be ignored. Only exported_model/ is included for a stable demo.

- DVC: keep .dvc/ committed (metadata); use remote storage (S3/GDrive) for large data if sharing or collaborating.

- Model registry: use MLflow registry for production flows; exported artifacts can be committed for demo or fetched from artifact stores in production.

- Reproducibility: pin major versions in requirements and use CI to validate builds; keep production container minimal (separate train vs. serve deps).

- Data realism: validate models on realistic holdout sets and guard against leakage, label drift, and unrealistic synthetic signals.

-------


📈 Future Improvements 

- Scheduled evaluation and drift detection (ZenML / MLflow monitors)

- Replace local exported artifact with remote artifact store + dynamic pull during deployment

- Add Prometheus/Grafana or hosted monitoring for latency & errors

- Add automated integration smoke tests in CI that spin up container and call /predict

- Use SQLite or managed DB for MLflow backend instead of filesystem for team collaboration

- Add data lineage & model cards / documented model assumptions

--------

📌 Author

**Manoj Mareedu**
Data Scientist / ML Engineer
GitHub: https://github.com/ManojMareedu
LinkedIn: https://www.linkedin.com/in/manojmareedu/
