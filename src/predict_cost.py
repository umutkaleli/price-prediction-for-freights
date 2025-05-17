import joblib
import pandas as pd

# Model yükle
model = joblib.load("models/rf_model.pkl")

def predict_cost(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return prediction
