from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Fraud Detection Model API")


MODEL_PATH = "exported_model"


# Load model once at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.sklearn.load_model(MODEL_PATH)

class Transaction(BaseModel):
    distance_from_home: float
    distance_from_last_transaction: float
    ratio_to_median_purchase_price: float
    repeat_retailer: float
    used_chip: float
    used_pin_number: float
    online_order: float


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):

    input_df = pd.DataFrame([transaction.dict()])

    prediction = model.predict(input_df)[0]

    # If model supports probability
    try:
        proba = model.predict_proba(input_df)[0][1]
    except Exception:
        proba = None

    return {
        "prediction": int(prediction),
        "probability": float(proba) if proba is not None else None
    }
