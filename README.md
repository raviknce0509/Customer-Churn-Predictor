#  Customer Churn Predictor

> End-to-end ML system predicting telecom customer churn with XGBoost (ROC-AUC: 0.83)

##  Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_HF_USERNAME/customer-churn-predictor)

##  Architecture
Raw Data → Feature Engineering → XGBoost → FastAPI → Docker → HF Spaces

### Production Architecture (AWS)
Raw Data → S3 → Feature Engineering → XGBoost → API Gateway → Lambda → CloudWatch

##  Model Performance
| Metric | Score |
|---|---|
| ROC-AUC | 0.83 |
| Precision | 0.71 |
| Recall | 0.68 |

##  Tech Stack
Python · XGBoost · Scikit-learn · FastAPI · Docker · Gradio · Hugging Face Spaces

##  Run Locally
```bash
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload
```
