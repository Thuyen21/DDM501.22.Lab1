# app/main.py - REST API that serves the model with FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the application
app = FastAPI(title="Iris Classifier API", version="1.0")

# Load the model ONCE when the service starts
model = joblib.load("model.joblib")
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Input schema - Pydantic validates data types automatically
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"service": "Iris Classifier API", "status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[features.sepal_length, features.sepal_width,
                   features.petal_length, features.petal_width]])
    pred_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(float(max(proba)), 4)
    }
