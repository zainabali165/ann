import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🎓 ANN Student Performance Evaluator")

st.write("Enter student details:")

attendance = st.slider("Attendance (%)", 0, 100)
assignment = st.slider("Assignment Marks", 0, 100)
quiz = st.slider("Quiz Marks", 0, 100)
mid = st.slider("Mid Exam Marks", 0, 100)
study_hours = st.slider("Study Hours per Day", 0, 12)

if st.button("Predict Performance"):
    data = np.array([[attendance, assignment, quiz, mid, study_hours]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    st.success(f"Predicted Result: {result[0]}")

    # Simple interpretation
    if result[0] == 1:
        st.info("This student is likely to PASS ✅")
    else:
        st.warning("This student may FAIL ❌")