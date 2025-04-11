import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Load dataset
data_path = 'creditcard.csv'
df = pd.read_csv(data_path)

# Assume last column is target, adjust accordingly
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize MLflow
mlflow.set_experiment("mlaas_model_comparison")

best_model = None
best_score = 0
best_model_name = ""

models = {
    "SVM": SVC(probability=True),
    "LightGBM": LGBMClassifier()
}

for model_name, model in models.items():
    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Log metrics
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Save best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_model_name = model_name
            joblib.dump(model, "best_model.pkl")

# Save best model
if best_model:
    mlflow.sklearn.log_model(best_model, "best_model")
    print(f"Best model: {best_model_name} with Accuracy: {best_score}")

# Deployment script
import fastapi
from fastapi import FastAPI
import uvicorn

app = FastAPI()
model = joblib.load("best_model.pkl")

@app.post("/predict/")
def predict(features: list):
    prediction = model.predict([np.array(features)])
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
