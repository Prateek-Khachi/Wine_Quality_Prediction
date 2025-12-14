import joblib
import numpy as np

# Load models
classifier = joblib.load("models/classifier.pkl")
regressor = joblib.load("models/regressor.pkl")
features_order = joblib.load("models/features.pkl")

def predict(input_dict, mode="classification"):
    """
    input_dict: dict {feature_name: value}
    mode: 'classification' or 'regression'
    """

    # Arrange features in correct order
    X = np.array([[input_dict[f] for f in features_order]])

    if mode == "classification":
        pred = classifier.predict(X)[0]
        return int(pred)

    elif mode == "regression":
        pred = regressor.predict(X)[0]
        return round(float(pred), 2)

    else:
        raise ValueError("Invalid prediction mode")
