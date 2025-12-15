from flask import Flask, render_template, request
from ml.predictor import predict, get_feature_importance
import matplotlib
matplotlib.use("Agg")  # IMPORTANT: non-GUI backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# ==================================================
# 🔹 DIRECTORY SETUP (DO NOT HARDCODE STATIC PATH)
# ==================================================

# THIS LINE AUTOMATICALLY POINTS TO THE FOLDER
# WHERE app.py EXISTS (i.e., wine_app/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# THIS LINE POINTS TO wine_app/static/
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure static folder exists
os.makedirs(STATIC_DIR, exist_ok=True)

# ==================================================

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    mode = "regression"
    chart_path = None

    # Default input values
    values = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11,
        "total sulfur dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    if request.method == "POST":
        mode = request.form["mode"]

        for f in FEATURES:
            values[f] = float(request.form[f])

        input_data = [values[f] for f in FEATURES]
        result = predict(input_data, mode)

        # ================= FEATURE IMPORTANCE =================
        importance = get_feature_importance(mode)
        print("DEBUG: feature importance =", importance)

        if importance is not None and len(importance) > 0:
            filename = f"importance_{mode}.png"

            # FULL PATH TO SAVE IMAGE
            full_path = os.path.join(STATIC_DIR, filename)

            plt.figure(figsize=(8, 5))
            plt.barh(list(importance.keys()), list(importance.values()))
            plt.xlabel("Importance")
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(full_path)
            plt.close()

            chart_path = filename
            print("Graph saved at:", full_path)
        else:
            print("Feature importance not available")

    return render_template(
        "index.html",
        result=result,
        mode=mode,
        values=values,
        chart_path=chart_path,
        show_modal=True if result else False
    )

if __name__ == "__main__":
    app.run(debug=True)
