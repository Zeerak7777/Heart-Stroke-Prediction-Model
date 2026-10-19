
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("Scaler.pkl")
expected_columns = joblib.load("columns.pkl")
if callable(expected_columns):
    expected_columns = expected_columns()
st.title("Heart Stroke Prediction by ZEERAK")
st.markdown("Provide the following details")

age = st.slider("AGE", 18, 100, 40)
sex = st.selectbox("SEX", ["MALE", "FEMALE"])
chest_pain = st.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"]
)
resting_bp = st.number_input("Resting BP", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox(
    "Resting ECG", ["Normal", "ST", "LVH"]
)
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox(
    "Exercise induced - Angina", ["YES", "NO"]
)
oldpeak = st.slider("ST depression", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST-Slope", ["UP", "Flat", "Down"])
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

if st.button("Predict"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in the correct order
    input_df = input_df[expected_columns]

    # Scale input data
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
