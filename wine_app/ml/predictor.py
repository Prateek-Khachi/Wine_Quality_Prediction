import joblib
import numpy as np

# ---------------------------------
# Load models & metadata
# ---------------------------------
classifier = joblib.load("models/classifier.pkl")
regressor = joblib.load("models/regressor.pkl")
features_order = joblib.load("models/features.pkl")

# ---------------------------------
# Helper: Quality label
# ---------------------------------
def quality_label(value):
    if value <= 5:
        return "Poor 🍷"
    elif value == 6:
        return "Average 🙂"
    else:
        return "Good ⭐"

# ---------------------------------
# Prediction
# ---------------------------------
def predict(input_data, mode="regression"):
    """
    input_data: list of feature values (length = 11)
    mode: 'regression' or 'classification'
    """

    X = np.array(input_data).reshape(1, -1)

    if mode == "classification":
        pred = classifier.predict(X)[0]
        prob = max(classifier.predict_proba(X)[0])

        return {
            "prediction": int(pred),
            "label": quality_label(pred),
            "confidence": round(prob * 100, 2),
            "model": "Classification"
        }

    elif mode == "regression":
        pred = regressor.predict(X)[0]

        return {
            "prediction": round(float(pred), 2),
            "label": quality_label(round(pred)),
            "model": "Regression"
        }

    else:
        raise ValueError("Invalid mode")

# ---------------------------------
# Feature importance
# ---------------------------------
def get_feature_importance(mode="regression"):
    model = regressor if mode == "regression" else classifier

    if not hasattr(model, "feature_importances_"):
        return None

    return dict(zip(features_order, model.feature_importances_))
