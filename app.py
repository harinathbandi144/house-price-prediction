import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import joblib

import streamlit as st
st.write("✅ App started successfully")

# Load your trained model (make sure you saved it earlier with joblib.dump)
model = joblib.load("xgb_model.pkl")

st.title("🏠 House Price Prediction App")

st.write("Enter the house features below to predict the price:")

# Input fields for each feature
CRIM = st.number_input("CRIM (per capita crime rate)", value=0.0)
ZN = st.number_input("ZN (residential land zone %)", value=0.0)
INDUS = st.number_input("INDUS (non-retail business acres)", value=0.0)
CHAS = st.selectbox("CHAS (Charles River dummy)", [0, 1])
NOX = st.number_input("NOX (nitric oxide concentration)", value=0.0)
RM = st.number_input("RM (average number of rooms)", value=5.0)
AGE = st.number_input("AGE (proportion of old houses)", value=50.0)
DIS = st.number_input("DIS (distance to employment centers)", value=0.0)
RAD = st.number_input("RAD (accessibility to highways)", value=1)
TAX = st.number_input("TAX (property tax rate)", value=200)
PTRATIO = st.number_input("PTRATIO (pupil-teacher ratio)", value=15.0)
B = st.number_input("B (proportion of Black population)", value=300.0)
LSTAT = st.number_input("LSTAT (lower status %)", value=10.0)

# Predict button
if st.button("Predict Price"):
    input_data = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
                              columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"])
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${prediction*1000:.2f}")
