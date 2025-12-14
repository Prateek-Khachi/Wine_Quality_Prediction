import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

# Load dataset (comma separated)
df = pd.read_csv("data/winequality.csv")

print("Columns:", df.columns)

X = df.drop("quality", axis=1)
y = df["quality"]

# Save feature order (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "models/features.pkl")

# Train classifier
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
clf.fit(X, y)

joblib.dump(clf, "models/classifier.pkl")

print("✅ Classification model saved")
