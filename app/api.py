from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("models/best_model_xgboost.pkl")

# Eğitimde kullanılan OneHotEncoder'ı da kaydettiysen yükle
encoder = joblib.load("models/encoder.pkl")
expected_columns = joblib.load("models/expected_columns.pkl")  # X_final.columns

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # Feature Engineering
    df['load_ratio'] = df['cargo_ton'] / df['optimal_load_ton']
    df['overload_penalty'] = np.maximum(0, df['cargo_ton'] - df['max_legal_ton'])

    # distance_category
    df['distance_category'] = pd.cut(
        df['distance_km'], bins=[0, 300, 800, np.inf], labels=['short', 'medium', 'long']
    )

    # Kategorik ve sayısal ayrımı
    categorical_cols = ['country', 'vehicle', 'complexity_factor', 'distance_category']
    numeric_cols = ['cargo_ton', 'distance_km', 'duration_hr', 'optimal_load_ton', 'max_legal_ton', 'load_ratio', 'overload_penalty']

    # One-hot encode
    X_cat = encoder.transform(df[categorical_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))

    # Sayısallar
    X_num = df[numeric_cols].reset_index(drop=True)

    # Final veri seti
    X_final = pd.concat([X_num, X_cat_df], axis=1)

    # Eksik sütun varsa sıfırla (önlem)
    for col in expected_columns:
        if col not in X_final.columns:
            X_final[col] = 0
    X_final = X_final[expected_columns]  # doğru sıraya sok

    # Tahmin
    pred = float(model.predict(X_final)[0])
    lower = float(pred * 1.2)
    upper = float(pred * 1.4)

    return jsonify({
        "prediction": round(pred, 2),
        "range": [round(lower, 2), round(upper, 2)]
    })

if __name__ == "__main__":
    app.run(debug=True)
