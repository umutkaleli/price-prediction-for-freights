from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("models/rf_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]

    # Tahmin aralığı hesapla (basit ±%10)
    lower = pred * 0.9
    upper = pred * 1.1

    return jsonify({
        "prediction": round(pred, 2),
        "range": [round(lower, 2), round(upper, 2)]
    })

if __name__ == "__main__":
    app.run(debug=True)
