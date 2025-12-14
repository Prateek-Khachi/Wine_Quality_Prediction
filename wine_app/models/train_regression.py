import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

os.makedirs("models", exist_ok=True)

# Load dataset (comma separated)
df = pd.read_csv("data/winequality.csv")

X = df.drop("quality", axis=1)
y = df["quality"]

# Train regressor
reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
reg.fit(X, y)

joblib.dump(reg, "models/regressor.pkl")

print("✅ Regression model saved")
