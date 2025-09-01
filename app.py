import streamlit as st
import pickle
import pandas as pd

# Load the trained model and encoder
with open("model.pkl", "rb") as f:
    saved_objects = pickle.load(f)

model = saved_objects["model"]
label_encoder = saved_objects["label_encoder"]

# Streamlit app
st.title("🍽️ Meal Prediction App")

st.write("Predict the number of meals consumed based on attendance, day of week, and weather.")

# User inputs
attendance = st.number_input("Attendance", min_value=0, step=1)
day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
weather = st.selectbox("Weather", label_encoder.classes_)

# Encode categorical features
day_mapping = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
day_encoded = day_mapping[day_of_week]
weather_encoded = label_encoder.transform([weather])[0]

# Prediction
if st.button("Predict Meals Consumed"):
    input_df = pd.DataFrame([[attendance, day_encoded, weather_encoded]],
                            columns=["Attendance", "DayOfWeek", "Weather"])
    prediction = model.predict(input_df)[0]
    st.success(f"🍴 Predicted Meals Consumed: {round(prediction)}")



