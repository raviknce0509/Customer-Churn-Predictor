# Customer Churn Predictor

End-to-end ML system predicting telecom customer churn with 82%+ ROC-AUC. Deployed as a REST API on AWS EC2.

---

## Architecture

```
Raw Data → Feature Engineering → XGBoost Model → FastAPI → Docker → AWS EC2
```

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.83 |
| Precision (Churn) | 0.71 |
| Recall (Churn) | 0.68 |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data Processing | Python, pandas, scikit-learn |
| Model | XGBoost |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Deployment | AWS EC2 |
| EDA | Jupyter, Matplotlib, Seaborn |

---

## Project Structure

```
churn-predictor/
│
├── data/
│   └── telco_churn.csv          # dataset (add manually)
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # feature engineering
│   ├── train.py                 # model training
│   └── predict.py               # prediction logic
│
├── app/
│   └── main.py                  # FastAPI app
│
├── models/
│   └── churn_model.pkl          # saved model (generated)
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it at `data/telco_churn.csv`.

### 3. Train the model

```bash
python src/train.py
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

Swagger UI: `http://localhost:8000/docs`

---

## Docker

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

---

## API Endpoint

**POST** `/predict` — Returns churn probability and risk level

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

**Response:**

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7876,
  "risk_level": "HIGH"
}
```
