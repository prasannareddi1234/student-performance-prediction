import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load saved objects
# -------------------------------

with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)


with open("poly_model.pkl", "rb") as f:
    model = pickle.load(f)

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("Student Performance Prediction")
st.write("Enter student details to predict exam score")

hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=8)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=0)
physical_activity = st.slider("Physical Activity Level", min_value=0, max_value=6, value=3)

motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
gender = st.selectbox("Gender", ["Female", "Male"])


# -------------------------------
# Manual Encoding
# -------------------------------

motivation_low = 1 if motivation_level == "Low" else 0
motivation_medium = 1 if motivation_level == "Medium" else 0
gender_male = 1 if gender == "Male" else 0

physical_1 = 1 if physical_activity == 1 else 0
physical_2 = 1 if physical_activity == 2 else 0
physical_3 = 1 if physical_activity == 3 else 0
physical_4 = 1 if physical_activity == 4 else 0
physical_5 = 1 if physical_activity == 5 else 0
physical_6 = 1 if physical_activity == 6 else 0

# Set other dummies to 0 for simplicity
parental_low = 0
parental_medium = 0
access_low = 0
access_medium = 0
extracurricular_yes = 0
internet_yes = 0
family_low = 0
family_medium = 0
teacher_low = 0
teacher_medium = 0
school_public = 0
peer_neutral = 0
peer_positive = 0
learning_yes = 0
education_high = 0
education_post = 0
distance_moderate = 0
distance_near = 0


# -------------------------------
# Prepare input data
# -------------------------------

input_data = np.array([[
    hours_studied,
    attendance,
    sleep_hours,
    previous_scores,
    tutoring_sessions,
    parental_low,
    parental_medium,
    access_low,
    access_medium,
    extracurricular_yes,
    motivation_low,
    motivation_medium,
    internet_yes,
    family_low,
    family_medium,
    teacher_low,
    teacher_medium,
    school_public,
    peer_neutral,
    peer_positive,
    physical_1,
    physical_2,
    physical_3,
    physical_4,
    physical_5,
    physical_6,
    learning_yes,
    education_high,
    education_post,
    distance_moderate,
    distance_near,
    gender_male
]])

# Apply polynomial features
input_data = poly.transform(input_data)

# Apply scaling if scaler exists
if scaler is not None:
    input_data = scaler.transform(input_data)


# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Exam Score"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
