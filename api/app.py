from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")  # Make sure you saved this during training

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def read_root():
    return {"status": "App is running!", "message": "Welcome to Credit Card fraud detection!"}

@app.post("/predict/")
def predict(data: list[Transaction]):
    # Convert input list of Transaction to DataFrame
    df = pd.DataFrame([d.dict() for d in data])
    
    # Scale the data
    transformed = scaler.transform(df)
    
    # Make predictions
    prediction = model.predict(transformed)
    return {"prediction": prediction.tolist()}
