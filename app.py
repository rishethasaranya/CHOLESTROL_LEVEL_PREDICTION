import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load Model & Scaler
# -----------------------------
with open("cholesterol_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="Cholesterol Prediction App")
st.title("🫀 Cholesterol Level Prediction")
st.write("Enter patient details to predict cholesterol level")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (0 = No, 1 = Yes)", [0, 1])
restecg = st.number_input("Rest ECG (0-2)", 0, 2)
thalach = st.number_input("Max Heart Rate", 60, 220)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, step=0.1)
slope = st.number_input("Slope (0-2)", 0, 2)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4)
thal = st.number_input("Thal (0-3)", 0, 3)
num = st.number_input("Disease Status (0-4)", 0, 4)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Cholesterol"):

    input_data = np.array([[age, sex, cp, trestbps, fbs, restecg,
                            thalach, exang, oldpeak, slope,
                            ca, thal, num]])

    # Apply Scaling
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Cholesterol Level: {prediction[0]:.2f}")
