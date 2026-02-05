import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/model.pkl")

st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴")

st.title("😴 Sleep Quality Predictor")
st.write("Enter your details to predict sleep quality")

# ---------- User Inputs ----------
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0)
physical_activity = st.slider("Physical Activity Level", 0, 100, 50)
stress_level = st.slider("Stress Level", 0, 10, 5)
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
heart_rate = st.number_input("Heart Rate", 40, 120, 75)
daily_steps = st.number_input("Daily Steps", 1000, 20000, 8000)

# ---------- Encoding ----------
gender = 1 if gender == "Male" else 0

if bmi_category == "Normal":
    bmi_category = 0
elif bmi_category == "Overweight":
    bmi_category = 1
else:
    bmi_category = 2

# ---------- Prediction ----------
if st.button("Predict Sleep Quality"):
    input_data = np.array([[gender, age, sleep_duration,
                            physical_activity, stress_level,
                            bmi_category, heart_rate, daily_steps]])

    prediction = model.predict(input_data)[0]

    if prediction >= 8:
        st.success(f"🌟 Excellent Sleep Quality ({prediction})")
    elif prediction >= 6:
        st.info(f"🙂 Good Sleep Quality ({prediction})")
    else:
        st.warning(f"⚠️ Poor Sleep Quality ({prediction})")


