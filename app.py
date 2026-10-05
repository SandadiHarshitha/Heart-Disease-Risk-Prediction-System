import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load model and scaler

model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# Page Title

st.title("❤️ Heart Attack Prediction using Machine Learning")

st.write(
    "This application predicts heart disease risk based on patient health parameters "
    "using a Random Forest Machine Learning model."
)


st.subheader("Enter Patient Details")


# User Inputs

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=50
)


sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)


cp = st.selectbox(
    "Chest Pain Type",
    [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ]
)


trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50.0,
    max_value=250.0,
    value=120.0
)


chol = st.number_input(
    "Cholesterol Level",
    min_value=50.0,
    max_value=600.0,
    value=200.0
)


fbs = st.selectbox(
    "Fasting Blood Sugar",
    [False, True]
)


restecg = st.selectbox(
    "Rest ECG",
    [
        "normal",
        "lv hypertrophy",
        "st-t abnormality"
    ]
)


thalch = st.number_input(
    "Maximum Heart Rate",
    min_value=50.0,
    max_value=250.0,
    value=150.0
)


exang = st.selectbox(
    "Exercise Induced Angina",
    [False, True]
)


oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)


slope = st.selectbox(
    "Slope",
    [
        "flat",
        "upsloping",
        "downsloping"
    ]
)


ca = st.number_input(
    "Number of Major Vessels",
    min_value=0,
    max_value=4,
    value=0
)


thal = st.selectbox(
    "Thal",
    [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
)



# Prediction

if st.button("Predict Heart Disease Risk"):


    # Creating input dataframe

    input_data = pd.DataFrame({

        "age":[age],
        "trestbps":[trestbps],
        "chol":[chol],
        "thalch":[thalch],
        "oldpeak":[oldpeak],
        "ca":[ca],

        "sex_Male":[1 if sex=="Male" else 0],

        "cp_atypical angina":[1 if cp=="atypical angina" else 0],
        "cp_non-anginal":[1 if cp=="non-anginal" else 0],
        "cp_typical angina":[1 if cp=="typical angina" else 0],

        "fbs_True":[1 if fbs else 0],

        "restecg_normal":[1 if restecg=="normal" else 0],
        "restecg_st-t abnormality":[1 if restecg=="st-t abnormality" else 0],

        "exang_True":[1 if exang else 0],

        "slope_flat":[1 if slope=="flat" else 0],
        "slope_upsloping":[1 if slope=="upsloping" else 0],

        "thal_normal":[1 if thal=="normal" else 0],
        "thal_reversable defect":[1 if thal=="reversable defect" else 0]

    })


    # Scaling

    scaled_input = scaler.transform(input_data)


    # Prediction

    prediction = model.predict(scaled_input)

    probability = model.predict_proba(scaled_input)

    confidence = np.max(probability) * 100



    # Result

    if prediction[0] == 1:

        st.error("⚠️ High Risk of Heart Disease")

    else:

        st.success("✅ Low Risk of Heart Disease")


    st.info(
        f"Prediction Confidence: {confidence:.2f}%"
    )



    # Explainable AI

    st.subheader("🔍 Prediction Explanation")


    st.write(
        "Important factors considered by the model:"
    )


    if age > 50:
        st.write("• Age is a significant heart disease risk factor")


    if chol > 240:
        st.write("• High cholesterol level may increase risk")


    if trestbps > 140:
        st.write("• High resting blood pressure detected")


    if oldpeak > 2:
        st.write("• Higher ST depression value detected")


    if exang:
        st.write("• Exercise induced angina detected")


    if thalch < 120:
        st.write("• Lower maximum heart rate detected")


    if ca > 0:
        st.write("• Presence of major vessel blockage indicator")



    st.subheader("📊 Model Interpretability")


    st.write(
        """
        The Random Forest model mainly learns patterns from:

        • Cholesterol
        • Maximum heart rate
        • Age
        • Oldpeak
        • Exercise induced angina
        • Resting blood pressure

        These features were identified using feature importance analysis.
        """
    )