import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Yeni artırılmış veri setini yükle
df = pd.read_csv("data/augmented_dataset.csv")
X = df.drop(columns=["cost"])
y = df["cost"]

categorical_cols = ["start_country", "end_country", "kasa_tipi"]

# Pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Eğitim ve model kaydı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "models/rf_model.pkl")

print("Model başarıyla eğitildi ve kaydedildi.")
