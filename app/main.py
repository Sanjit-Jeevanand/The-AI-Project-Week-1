from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Default Predictor")

model = joblib.load("artifacts/model.joblib")

class LoanRequest(BaseModel):
    age: int
    income: int
    employment_years: float
    loan_amount: int
    interest_rate: float
    credit_score: int
    home_ownership: str
    loan_purpose: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: LoanRequest):
    data = pd.DataFrame([req.model_dump()])
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]
    return {
        "prediction": int(pred),
        "default_probability": float(proba)
    }