import streamlit as st
from ml.predictor import predict

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

st.title("🍷 Wine Quality Prediction WebApp")

mode = st.radio(
    "Choose Prediction Type",
    ["Classification", "Regression"]
)

st.subheader("Enter Wine Properties")

input_data = {}

input_data["fixed acidity"] = st.number_input("Fixed Acidity", 0.0, 20.0)
input_data["volatile acidity"] = st.number_input("Volatile Acidity", 0.0, 2.0)
input_data["citric acid"] = st.number_input("Citric Acid", 0.0, 1.0)
input_data["residual sugar"] = st.number_input("Residual Sugar", 0.0, 15.0)
input_data["chlorides"] = st.number_input("Chlorides", 0.0, 0.5)
input_data["free sulfur dioxide"] = st.number_input("Free Sulfur Dioxide", 0, 100)
input_data["total sulfur dioxide"] = st.number_input("Total Sulfur Dioxide", 0, 300)
input_data["density"] = st.number_input("Density", 0.990, 1.005)
input_data["pH"] = st.number_input("pH", 2.5, 4.5)
input_data["sulphates"] = st.number_input("Sulphates", 0.0, 2.0)
input_data["alcohol"] = st.number_input("Alcohol", 8.0, 15.0)

if st.button("Predict Wine Quality"):
    result = predict(
        input_data,
        mode="classification" if mode == "Classification" else "regression"
    )

    if mode == "Classification":
        st.success(f"🍷 Predicted Quality Class: {result}")
    else:
        st.success(f"🍷 Predicted Quality Score: {result}")
