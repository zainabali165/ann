import joblib
import numpy as np

# Load saved files
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    data = np.array([[attendance, assignment, quiz, mid, study_hours]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction[0]

# Test example
if __name__ == "__main__":
    result = evaluate_student(85, 80, 75, 70, 6)
    print("Predicted Result:", result)