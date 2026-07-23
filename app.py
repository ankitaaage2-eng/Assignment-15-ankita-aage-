
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("ford_model.pkl")
le_model = joblib.load("le_model.pkl")
le_trans = joblib.load("le_trans.pkl")
le_fuel = joblib.load("le_fuel.pkl")

st.set_page_config(page_title="Ford Car Price Prediction", layout="centered")

st.title("Ford Car Price Prediction")

car_model = st.selectbox("Car Model", le_model.classes_)
year = st.number_input("Year", 1996, 2026, 2018)
transmission = st.selectbox("Transmission", le_trans.classes_)
mileage = st.number_input("Mileage", value=10000)
fuel = st.selectbox("Fuel Type", le_fuel.classes_)
tax = st.number_input("Tax", value=150)
mpg = st.number_input("MPG", value=50.0)
engine = st.number_input("Engine Size", value=1.5)

if st.button("Predict Price"):
    df = pd.DataFrame({
        "model":[le_model.transform([car_model])[0]],
        "year":[year],
        "transmission":[le_trans.transform([transmission])[0]],
        "mileage":[mileage],
        "fuelType":[le_fuel.transform([fuel])[0]],
        "tax":[tax],
        "mpg":[mpg],
        "engineSize":[engine]
    })

    prediction = model.predict(df)[0]
    st.success(f"Predicted Price: £{prediction:.2f}")
