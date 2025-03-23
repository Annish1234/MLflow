# from fastapi import FastAPI
# import joblib
# import numpy as np

# app = FastAPI()
# model = joblib.load("model/model.pkl")

# @app.post("/predict/")
# def predict(features: list):
#     prediction = model.predict([np.array(features)])
#     return {"prediction": prediction.tolist()}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model/model.pkl")
print("Classes:", model.classes_)

#  new comment from github 2
class InputData(BaseModel):
    features: list[float]  # Ensures 'features' is a required list of floats
    #changes

@app.post("/predict/")
def predict(data: InputData):
    prediction = model.predict([np.array(data.features)])
    return {"prediction": prediction.tolist()}
