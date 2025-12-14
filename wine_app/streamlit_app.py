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
    Predict wine quality using **Machine Learning**.
    
    - **Regression** → predicts an exact quality score  
    - **Classification** → predicts a quality class with confidence  
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
# Input section
# --------------------------------------------------
st.subheader("🔢 Enter Wine Chemical Properties")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.0, 1.9)
    chlorides = st.slider("Chlorides", 0.01, 0.3, 0.076)

with col2:
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 80, 11)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 300, 34)
    density = st.slider("Density", 0.990, 1.005, 0.9978)
    ph = st.slider("pH", 2.8, 4.2, 3.51)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.56)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 9.4)

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
# Input warnings (soft validation)
# --------------------------------------------------
st.markdown("---")
st.subheader("⚠️ Input Validation")

warnings = []

if alcohol < 9:
    warnings.append("Low alcohol may result in lower quality.")
if volatile_acidity > 1.0:
    warnings.append("High volatile acidity often reduces wine quality.")
if density > 1.000:
    warnings.append("High density usually indicates lower alcohol.")

if warnings:
    for w in warnings:
        st.warning(w)
else:
    st.success("Inputs look reasonable.")

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
            f"🏷 Interpreted Category: {result['label']}"
        )

    # --------------------------------------------------
    # Feature importance
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
