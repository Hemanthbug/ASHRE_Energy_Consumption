import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

print("Loading parquet file...")

df = pd.read_parquet("data/processed/clean_data.parquet")

# Drop nulls just in case
df = df.dropna()

print("Total rows:", len(df))

# 🚀 DEVELOPMENT MODE: sample to reduce training time
# You can increase later for final training
df = df.sample(n=200000, random_state=42)

print("Training rows after sampling:", len(df))

# Feature set (including time features)
X = df[
    [
        "building_id",
        "meter",
        "square_feet",
        "air_temperature",
        "cloud_coverage",
        "hour",
        "month",
        "dayofweek"
    ]
]

y = df["meter_reading"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Faster & optimized model
model = RandomForestRegressor(
    n_estimators=25,     # reduced trees
    max_depth=15,        # limit depth for speed
    n_jobs=-1,           # use all CPU cores
    random_state=42
)

model.fit(X_train, y_train)

print("Evaluating model...")

preds = model.predict(X_test)

mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

print("RMSE (log scale):", rmse)

# Save model
joblib.dump(model, "models/energy_model.pkl")

print("Model saved 🚀")