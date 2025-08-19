import streamlit as st
import pandas as pd
import numpy as np

# Try loading model using joblib first, then fallback to pickle
try:
    import joblib
    model = joblib.load("model.pkl")
except:
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

st.title("🍲 Food Waste Prediction App")
st.write("Predict food demand in hostels/cafeterias to reduce waste.")

# Input fields
attendance = st.number_input("Student/Employee Attendance", min_value=0, max_value=1000, value=200)
day_of_week = st.selectbox("Day of the Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])

# Encode categorical inputs
day_mapping = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
weather_mapping = {"Sunny":0,"Rainy":1,"Cloudy":2}

day_val = day_mapping[day_of_week]
weather_val = weather_mapping[weather]

# Predict button
if st.button("Predict Food Requirement"):
    input_data = np.array([[attendance, day_val, weather_val]])
    prediction = model.predict(input_data)[0]
    st.success(f"🍛 Predicted Meals Needed: {int(prediction)}")
