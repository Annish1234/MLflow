# def test_placeholder():
#     assert True 
# #test case modified1

import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.app import app

client = TestClient(app)

sample_input = [
    {
        "Time": 151029.0,
        "V1": -3.8182137183,
        "V2": 2.5513376778,
        "V3": -4.7591583159,
        "V4": 1.6369669588,
        "V5": -1.1679000839,
        "V6": -1.6784133878,
        "V7": -3.1447324721,
        "V8": 1.2451055504,
        "V9": -1.6925413541,
        "V10": -4.7599313739,
        "V11": 3.6422569394,
        "V12": -4.1652023226,
        "V13": 0.7030698751,
        "V14": -7.624316314,
        "V15": -1.4980123996,
        "V16": -4.07999233,
        "V17": -6.7171765476,
        "V18": -1.8875496491,
        "V19": 1.3567482897,
        "V20": 0.164453446,
        "V21": 0.8376854401,
        "V22": 0.7617121271,
        "V23": -0.4176942659,
        "V24": -0.4697124715,
        "V25": -0.2259342358,
        "V26": 0.5864152485,
        "V27": -0.3481074059,
        "V28": 0.0877768724,
        "Amount": 10.7
    }
]

sample_input_2 = [
  {
    "Time": 12345.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62
  }
]

# 1. Valid prediction test

def test_predict_endpoint():
    response = client.post("/predict/", json=[sample_input[0]])
    print("Response JSON:", response.json())  
    assert response.status_code == 200
    assert "prediction" in response.json()

#  2. Test empty input
def test_predict_empty_input():
    response = client.post("/predict/", json=[])
    print("Empty Input Response:", response.json())
    assert response.status_code in [400, 422]

#  3. Missing field test
def test_predict_missing_field():
    incomplete_input = sample_input[0].copy()
    del incomplete_input["V1"]
    response = client.post("/predict/", json=[incomplete_input])
    print("Missing Field Response:", response.json())
    assert response.status_code == 422

#  4. Invalid data type test
def test_predict_invalid_type():
    invalid_input = sample_input[0].copy()
    invalid_input["Amount"] = "invalid"
    response = client.post("/predict/", json=[invalid_input])
    print("Invalid Type Response:", response.json())
    assert response.status_code == 422

#  5. GET root endpoint test
def test_read_root():
    response = client.get("/")
    print("Root Response:", response.json())
    assert response.status_code == 200
    assert response.json()["status"] == "App is running!"

#  6. Multiple records test
def test_predict_multiple_inputs():
    inputs = [sample_input[0], sample_input_2[0]]
    response = client.post("/predict/", json=inputs)
    print("Multiple Input Response:", response.json())
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], list)
    assert len(response.json()["prediction"]) == 2