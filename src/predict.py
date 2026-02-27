"""
Prediction logic for the Telco Churn Predictor.
Loads the saved model artifact and runs inference.
"""

import pickle
from pathlib import Path

import pandas as pd

from src.preprocess import clean_data, encode_features, scale_features

DEFAULT_MODEL_PATH = "models/churn_model.pkl"


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """Load the pickled model artifact (model + scaler + feature_names)."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    return artifact


def predict_single(customer: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Predict churn probability for a single customer.

    Args:
        customer: dict of raw feature values (mirrors CSV columns).
        model_path: path to the saved model artifact.

    Returns:
        dict with keys 'churn_prediction' (0/1) and 'churn_probability' (float).
    """
    artifact = load_model(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    df = pd.DataFrame([customer])
    df = clean_data(df)
    df = encode_features(df)
    df, _ = scale_features(df, scaler=scaler, fit=False)

    # Align columns to training feature set
    df = df.reindex(columns=feature_names, fill_value=0)

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "label": "Churn" if prediction == 1 else "No Churn",
    }


def predict_batch(filepath: str, model_path: str = DEFAULT_MODEL_PATH) -> pd.DataFrame:
    """
    Run predictions on a batch CSV file.

    Returns a DataFrame with original data + 'churn_prediction' and 'churn_probability' columns.
    """
    artifact = load_model(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    raw_df = pd.read_csv(filepath)
    df = clean_data(raw_df.copy())
    df = encode_features(df)
    df, _ = scale_features(df, scaler=scaler, fit=False)
    df = df.reindex(columns=feature_names, fill_value=0)

    raw_df["churn_prediction"] = model.predict(df)
    raw_df["churn_probability"] = model.predict_proba(df)[:, 1].round(4)

    return raw_df
