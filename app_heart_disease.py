import streamlit as st
import pandas as pd
import joblib
import os

# Load the saved best model
MODEL_PATH = r"C:\Users\devar\Downloads\heart_attack\best_heart_model.joblib"


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please train and save the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.title("❤️ Heart Disease Prediction (India)")
st.write("Enter patient details below to predict the presence of heart disease.")

# Input fields for each feature
age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
chestpain = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restingrelectro = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
maxheartrate = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exerciseangia = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3],
                     format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1])
noofmajorvessels = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])

# Prepare input DataFrame
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "chestpain": chestpain,
    "restingBP": restingBP,
    "serumcholestrol": serumcholestrol,
    "fastingbloodsugar": fastingbloodsugar,
    "restingrelectro": restingrelectro,
    "maxheartrate": maxheartrate,
    "exerciseangia": exerciseangia,
    "oldpeak": oldpeak,
    "slope": slope,
    "noofmajorvessels": noofmajorvessels
}])

# Predict button
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.write("### Prediction:", "Presence of Heart Disease" if prediction == 1 else "No Heart Disease")
    st.write(f"**Probability of Heart Disease:** {probability*100:.2f}%")
