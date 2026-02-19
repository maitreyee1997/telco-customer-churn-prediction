import joblib
import pandas as pd

MODEL_PATH = "models/churn_pipeline.pkl"

model = joblib.load(MODEL_PATH)


def predict_customer(input_dict: dict):
    df = pd.DataFrame([input_dict])
    prob = model.predict_proba(df)[0][1]
    return prob
