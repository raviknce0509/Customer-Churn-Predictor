""" 
Preprocessing module for the customer churn predictor project.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess(df: pd.DataFrame, fit: bool = True, 
               encoders: dict = None, scaler=None):
    df = df.copy()
    
    # Drop customer ID
    df.drop(columns=['customerID'], inplace=True)
    
    # Fix TotalCharges (comes in as string)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        joblib.dump(encoders, '../models/encoders.pkl')
    else:
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])
    
    # Scale numeric features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        joblib.dump(scaler, '../models/scaler.pkl')
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    
    return df, encoders, scaler