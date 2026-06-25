import os
import mlflow.pyfunc
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class PredictionRequest(BaseModel):
    state_tensor: List[List[List[float]]] # (29, 15, 12)
    mu_prior: List[float]                 # (29,)
    D_prior: List[List[float]]            # (29, 29)

model = None

@app.on_event("startup")
def load_model():
    global model
    model_uri = os.environ.get("MODEL_URI", "./model_artifacts")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    print(f"Loading MLflow model from {model_uri}...")
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")

@app.post("/predict")
def predict(request: PredictionRequest):
    input_dict = {
        "state_tensor": request.state_tensor,
        "mu_prior": request.mu_prior,
        "D_prior": request.D_prior
    }
    res = model.predict(input_dict)
    return {"refined_weights": res}

@app.get("/health")
def health():
    return {"status": "healthy"}
