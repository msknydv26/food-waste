
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("food_waste_data.csv")

st.title("🍲 Smart Canteen: Food Waste Prediction")
st.write("Predict meal demand & reduce food waste in hostels/corporate cafeterias.")

# Dashboard
st.subheader("📊 Data Overview")
st.dataframe(df.head())

# KPIs
avg_waste = df['FoodWasted'].mean()
st.metric("Average Daily Waste", f"{avg_waste:.1f} meals")

# Plot: Waste by Day
st.subheader("Food Waste by Day of Week")
waste_by_day = df.groupby("DayOfWeek")["FoodWasted"].mean()
st.bar_chart(waste_by_day)

# Model Training
X = pd.get_dummies(df[["DayOfWeek", "Weather", "Attendance"]], drop_first=True)
y = df["MealsCooked"]

model = LinearRegression()
model.fit(X, y)

# -------------------
# Prediction Tool
# -------------------
st.subheader("🔮 Predict Tomorrow's Meals")
day = st.selectbox("Day of Week", df["DayOfWeek"].unique())
weather = st.selectbox("Weather", df["Weather"].unique())
attendance = st.slider("Expected Attendance", 150, 300, 250)

# Prepare input
input_data = pd.DataFrame([[day, weather, attendance]], columns=["DayOfWeek", "Weather", "Attendance"])
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=X.columns, fill_value=0)

predicted_meals = model.predict(input_data)[0]
recommended = round(predicted_meals)

st.success(f"✅ Recommended Meals to Cook: {recommended}")
st.info(f"⚠️ Expected Waste: {recommended - attendance} meals (approx.)")

# -------------------
# 7-Day Forecast
# -------------------
st.subheader("📆 7-Day Forecast")

future_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
future_weather = ["Sunny","Rainy","Cloudy","Cold"]

forecast_data = []
for d in future_days:
    w = np.random.choice(future_weather)  # simulate random weather
    att = np.random.randint(200, 290)     # simulate future attendance
    temp_df = pd.DataFrame([[d, w, att]], columns=["DayOfWeek", "Weather", "Attendance"])
    temp_df = pd.get_dummies(temp_df)
    temp_df = temp_df.reindex(columns=X.columns, fill_value=0)
    pred_meals = model.predict(temp_df)[0]
    forecast_data.append([d, w, att, round(pred_meals), round(pred_meals - att)])

forecast_df = pd.DataFrame(forecast_data, columns=["DayOfWeek","Weather","Attendance","PredictedMeals","ExpectedWaste"])
st.dataframe(forecast_df)

# Plot forecast
st.subheader("📈 Forecasted Meals vs Attendance")
st.line_chart(forecast_df[["PredictedMeals","Attendance"]])
