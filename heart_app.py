import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("heart_model.pkl")
columns = joblib.load("heart_columns.pkl")

st.title("Heart Disease Prediction System")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["M", "F"])
chestpaintype = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
restingbp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
fastingbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restingecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.number_input("Maximum Heart Rate", value=150)
exerciseangina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", value=1.0)
stslope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Predict Button
if st.button("Predict"):

    sample_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chestpaintype,
        'RestingBP': restingbp,
        'Cholesterol': cholesterol,
        'FastingBS': fastingbs,
        'RestingECG': restingecg,
        'MaxHR': maxhr,
        'ExerciseAngina': exerciseangina,
        'Oldpeak': oldpeak,
        'ST_Slope': stslope
    }

    sample_df = pd.DataFrame([sample_data])

    # Encoding
    sample_df = pd.get_dummies(sample_df)

    # Match training columns
    sample_df = sample_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(sample_df)

    if prediction[0] == 1:
        st.error("Heart Disease: Yes")
    else:
        st.success("Heart Disease: No")