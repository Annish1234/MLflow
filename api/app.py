from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
from model.visualizations import generate_visual_report
from fastapi.responses import FileResponse

app = FastAPI()
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")  

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
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Input list is empty")
    # Convert input list of Transaction to DataFrame 
    #new change
    # df = pd.DataFrame([item.dict() for item in data])
    df = pd.DataFrame([item.model_dump() for item in data])
    transformed = scaler.transform(df)
    prediction = model.predict(transformed)
    return {"prediction": prediction.tolist()}

@app.get("/download-report")
def download_report():
    report_path = "visual_report.pdf"

    # Generate report if it doesn't exist
    if not os.path.exists(report_path):
        generate_visual_report()

    return FileResponse(path=report_path, filename="visual_report.pdf", media_type='application/pdf')