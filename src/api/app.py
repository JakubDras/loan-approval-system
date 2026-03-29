from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import mlflow.pytorch
import torch
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.preprocess import load_config, prepare_data

app = FastAPI(
    title="Loan Approval API",
    description="Low-Latency ML API do oceny zdolności kredytowej oparte na skompresowanym modelu Int8.",
    version="1.0.0"
)

model = None
scaler = None
expected_features = []

@app.on_event("startup")
def load_artifacts():
    global model, scaler, expected_features
    print("Inicjalizacja API: Pobieranie artefaktów...")

    config = load_config()
    _, _, _, _, _, _, fitted_scaler = prepare_data(config)
    scaler = fitted_scaler

    df = pd.read_csv(config['data']['raw_data_path'])
    df = df.drop("customer_id", axis=1)
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    features_to_drop = config['data']['features_to_drop']
    df = df.drop(columns=[c for c in features_to_drop if c in df.columns])
    expected_features = list(df.drop('loan_status', axis=1).columns)

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    runs = mlflow.search_runs(experiment_names=["Loan_Approval_Optimization"], order_by=["start_time DESC"])
    last_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{last_run_id}/production_model"

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    model.cpu()
    print(f"Gotowe! Model załadowany. Oczekuje {len(expected_features)} cech wejściowych.")

class LoanRequest(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(request: LoanRequest):
    try:
        if len(request.features) != len(expected_features):
            raise ValueError(f"Oczekiwano {len(expected_features)} cech, otrzymano {len(request.features)}.")

        input_array = np.array(request.features).reshape(1, -1)

        scaled_array = scaler.transform(input_array)
        input_tensor = torch.tensor(scaled_array, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()
            prediction = int(prob > 0.5)

        return {
            "approved": bool(prediction),
            "probability": round(prob, 4),
            "model_version": "int8_quantized"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))