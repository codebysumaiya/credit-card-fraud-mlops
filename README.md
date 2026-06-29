# Credit Card Fraud Detection — End-to-End MLOps Pipeline

A full MLOps pipeline for detecting fraudulent credit card transactions,
covering data versioning, preprocessing, model training with experiment
tracking, containerized deployment, CI/CD, and monitoring.

## Architecture

```
Data Collection -> Version Control -> Data Versioning (DVC)
      -> Data Processing -> Model Training
      -> Experiment Tracking (MLflow) -> Model Registry (MLflow)
      -> Containerization (Docker) -> Model Deployment (FastAPI)
      -> CI/CD (GitHub Actions) -> Monitoring (Prometheus + Grafana)
```

## Tech Stack

| Stage | Tool |
|---|---|
| Data versioning | DVC |
| Data processing | Python, pandas, imbalanced-learn |
| Model training | scikit-learn |
| Experiment tracking & registry | MLflow |
| API serving | FastAPI |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus, Grafana |

## Project Structure

```
credit-card-fraud-mlops/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── app/
│   ├── app.py                    # FastAPI server
│   ├── templates/index.html      # Frontend form
│   └── static/style.css
├── data/
│   ├── raw/creditcard.csv        # Raw Kaggle dataset (DVC-tracked)
│   └── processed/                # train.csv / test.csv (pipeline output)
├── models/
│   ├── fraud_model.pkl           # Trained model
│   └── metrics.json              # Evaluation metrics
├── monitoring/prometheus.yml
├── src/
│   ├── data_preprocessing.py
│   ├── data_training.py
│   └── predict.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_prediction.py
├── params.yaml                   # Central config
├── dvc.yaml                      # Pipeline stage definitions
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd credit-card-fraud-mlops
python -m venv venv
source venv/Scripts/activate    # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Get the data

Download the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place `creditcard.csv` in `data/raw/`.

### 3. Pull data via DVC (if already tracked by a teammate)

```bash
dvc pull
```

## Running the Pipeline

### Run everything via DVC (recommended)

```bash
dvc repro
```

### Or run each stage manually

```bash
python src/data_preprocessing.py
python src/data_training.py
```

### View experiment tracking

```bash
mlflow ui
# open http://127.0.0.1:5000
```

## Running the API

### Locally

```bash
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
# open http://127.0.0.1:8000
```

### Via Docker (API + Prometheus + Grafana together)

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Fraud detection app | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana (login: admin/admin) | http://localhost:3000 |

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD

Every push to `main` automatically runs the test suite and builds the
Docker image via GitHub Actions (`.github/workflows/ci.yml`). Check the
**Actions** tab on GitHub to see pipeline runs.

## Notes

- SMOTE balancing is applied only to the training set, never the test
  set, so evaluation metrics reflect real transaction data.
- Grafana's default `admin`/`admin` login is for local development only
  — change it before deploying anywhere network-reachable.