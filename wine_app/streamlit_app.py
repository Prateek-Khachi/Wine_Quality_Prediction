import streamlit as st
import matplotlib.pyplot as plt
from ml.predictor import predict, get_feature_importance

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction App")

st.write(
    """
    This app predicts **wine quality** using two ML models:
    - **Regression** → predicts exact quality score  
    - **Classification** → predicts quality class
    """
)

# --------------------------------------------------
# Model selection
# --------------------------------------------------
mode = st.radio(
    "Select Model Type",
    ("regression", "classification"),
    horizontal=True
)

st.markdown("---")

# --------------------------------------------------
# Input fields
# --------------------------------------------------
st.subheader("🔢 Enter Wine Chemical Properties")

fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.7)
citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=1.9)
chlorides = st.number_input("Chlorides", min_value=0.0, value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, value=34.0)
density = st.number_input("Density", min_value=0.0, value=0.9978, format="%.4f")
ph = st.number_input("pH", min_value=0.0, value=3.51)
sulphates = st.number_input("Sulphates", min_value=0.0, value=0.56)
alcohol = st.number_input("Alcohol", min_value=0.0, value=9.4)

input_data = [
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    ph,
    sulphates,
    alcohol
]

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.markdown("---")

if st.button("🔮 Predict Wine Quality"):
    result = predict(input_data, mode)

    if mode == "classification":
        st.success(
            f"🍷 Predicted Quality: {result['prediction']}\n"
            f"🏷 Category: {result['label']}\n"
            f"📊 Confidence: {result['confidence']}%"
        )
    else:
        st.success(
            f"🍷 Predicted Quality Score: {result['prediction']}\n"
            f"🏷 Category: {result['label']}"
        )


    # --------------------------------------------------
    # Feature Importance (OPTION 1)
    # --------------------------------------------------
    st.subheader("🔍 Feature Importance")

    importance = get_feature_importance(mode)

    if importance:
        features = list(importance.keys())
        values = list(importance.values())

        fig, ax = plt.subplots()
        ax.barh(features, values)
        ax.set_xlabel("Importance")
        ax.set_title(f"{mode.capitalize()} Model Feature Importance")

        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")
