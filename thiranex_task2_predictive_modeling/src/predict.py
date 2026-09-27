"""
Inference Script for Task 2: Predictive Modeling
Loads saved model pipeline from models/best_churn_model.joblib
and performs churn prediction on sample customer profile inputs.
"""

import os
import sys
import pandas as pd
import joblib

# Ensure project root is in sys.path for joblib unpickling of custom classes
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import CustomColumnTransformer
from src.train import MLPipeline, VectorizedLogisticRegression, VectorizedDecisionTree, VectorizedRandomForest, Node


def predict_churn(customer_data: dict, model_path: str = None) -> dict:
    """Predict churn probability and classification label for a customer dictionary."""
    if model_path is None:
        model_path = os.path.join(project_root, "models", "best_churn_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train model first!")

    model = joblib.load(model_path)
    df_input = pd.DataFrame([customer_data])

    churn_pred = model.predict(df_input)[0]
    churn_proba = model.predict_proba(df_input)[0][1]

    result = {
        "churn_prediction": "Churn (1)" if churn_pred == 1 else "Retained (0)",
        "churn_probability_pct": round(float(churn_proba) * 100, 2),
        "risk_level": "High" if churn_proba >= 0.6 else ("Medium" if churn_proba >= 0.3 else "Low")
    }

    return result


if __name__ == "__main__":
    # Sample Customer Profile (High Risk: Month-to-month, Fiber Optic, short tenure)
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
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
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.70,
        "TotalCharges": 171.40
    }

    print("==================================================")
    print("CUSTOMER CHURN INFERENCE DEMO")
    print("==================================================")
    print("Input Customer Profile:")
    for k, v in sample_customer.items():
        print(f" - {k:20s}: {v}")

    prediction = predict_churn(sample_customer)
    print("\nPrediction Output:")
    for k, v in prediction.items():
        print(f" - {k:25s}: {v}")
