import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Exam Score Predictor")
st.write("Enter student details to predict exam score")

attendance = st.slider("Attendance (%)", 50, 100, 75)
hours_studied = st.slider("Hours Studied per Day", 1, 10, 5)
previous_score = st.slider("Previous Exam Score", 40, 100, 70)
tutoring_sessions = st.slider("Tutoring Sessions per Month", 0, 10, 2)

access_resources = st.selectbox("High Access to Resources?", ["Yes", "No"])
parent_involvement = st.selectbox("High Parental Involvement?", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
school_type = st.selectbox("School Type", ["Public", "Private"])
sleep_hours = st.slider("Sleep Hours per Day", 4, 10, 7)

access_resources = 1 if access_resources == "Yes" else 0
parent_involvement = 1 if parent_involvement == "Yes" else 0
gender = 1 if gender == "Male" else 0
school_type = 1 if school_type == "Public" else 0

input_data = np.array([[
    attendance,
    hours_studied,
    previous_score,
    tutoring_sessions,
    access_resources,
    parent_involvement,
    gender,
    school_type,
    sleep_hours
]])

if st.button("Predict Exam Score"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")

# streamlit run Student_Project_App.py
