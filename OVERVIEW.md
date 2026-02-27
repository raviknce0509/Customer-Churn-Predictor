# Project Overview & Setup Guide

## What This Project Does

This project builds an end-to-end machine learning pipeline that predicts whether a telecom customer will churn. A trained XGBoost model is served via a FastAPI REST API, containerised with Docker, and deployable to AWS EC2.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 | via conda recommended |
| conda | any | for environment management |
| Docker Desktop | any | for containerised run |
| Git | any | to clone the repo |

---

## Step-by-Step Setup

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd Customer-Churn-Predictor
```

### Step 2 — Add the Dataset

Download the Telco Customer Churn dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the file at:
```
data/telco_churn.csv
```

### Step 3 — Create the Conda Environment

```bash
conda create -n churn-predictor python=3.11 -y
conda activate churn-predictor
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Train the Model

```bash
cd src
python train.py
```

This generates three files in `models/`:
```
models/
├── churn_model.pkl     # trained XGBoost model
├── encoders.pkl        # fitted LabelEncoders
└── scaler.pkl          # fitted StandardScaler
```

Expected output:
```
              precision    recall  f1-score   support
           0       0.84      0.91      0.87      1035
           1       0.66      0.51      0.57       374
    accuracy                           0.80      1409
ROC-AUC Score: 0.8434
✅ Model saved to models/churn_model.pkl
```

### Step 6 — Run the API

```bash
cd app
uvicorn main:app --reload
```

Or from the project root using the conda Python directly:
```bash
/opt/anaconda3/envs/churn-predictor/bin/uvicorn app.main:app --reload
```

API is live at: `http://127.0.0.1:8000`
Swagger UI at: `http://127.0.0.1:8000/docs`

---

## Running with Docker

### Build the Image

```bash
docker build -t churn-predictor .
```

### Run the Container

```bash
docker run -p 8000:8000 churn-predictor
```

### Run in Background (detached)

```bash
docker run -d -p 8000:8000 --name churn-api churn-predictor
```

### Useful Docker Commands

```bash
docker logs churn-api          # view server logs
docker stop churn-api          # stop the container
docker start churn-api         # restart the container
docker rm churn-api            # remove the container
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/
```

Response:
```json
{"message": "Churn Predictor API is live"}
```

### Predict Churn

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.45,
    "TotalCharges": 191.0
  }'
```

Response:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7876,
  "risk_level": "HIGH"
}
```

### Risk Level Thresholds

| risk_level | churn_probability |
|---|---|
| HIGH | > 0.70 |
| MEDIUM | 0.40 – 0.70 |
| LOW | < 0.40 |

---

## Running the Notebook (EDA)

```bash
conda activate churn-predictor
pip install jupyter matplotlib seaborn
cd notebooks
jupyter notebook 01_eda_and_modeling.ipynb
```

The notebook covers:
1. Data loading and inspection
2. Churn distribution and feature distributions
3. Correlation heatmap
4. Model training and evaluation
5. Confusion matrix
6. Feature importance (top 15)

---

## Project File Reference

| File | Purpose |
|---|---|
| `src/preprocess.py` | Cleans data, encodes categoricals, scales numerics |
| `src/train.py` | Trains XGBoost, saves model + encoders + scaler |
| `src/predict.py` | Single and batch prediction logic |
| `app/main.py` | FastAPI app with `/predict` and `/` endpoints |
| `models/churn_model.pkl` | Trained XGBoost model |
| `models/encoders.pkl` | Fitted LabelEncoders (one per categorical column) |
| `models/scaler.pkl` | Fitted StandardScaler for numeric features |
| `Dockerfile` | Builds the API container (Python 3.10-slim) |
| `requirements.txt` | All Python dependencies with pinned versions |

---

## Troubleshooting

**`FileNotFoundError: models/churn_model.pkl`**
→ Run `python src/train.py` first to generate the model files.

**`conda run` picks wrong Python version**
→ Use the direct binary: `/opt/anaconda3/envs/churn-predictor/bin/python`

**Docker build fails with numpy conflict**
→ Ensure `requirements.txt` has `numpy==1.26.4` (not `1.24.0` — excluded by seaborn).

**Port 8000 already in use**
→ Kill the existing process: `kill $(lsof -ti:8000)`
