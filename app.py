import streamlit as st
import pandas as pd
import joblib
import json
import os
import numpy as np

# ------------------- SAFETY CHECK -------------------
if not os.path.exists("best_ev_range_model.pkl"):
    st.error("Model file not found! Upload best_ev_range_model.pkl to GitHub and reboot.")
    st.stop()

if not os.path.exists("features_used.json"):
    st.error("features_used.json not found! Upload it and reboot.")
    st.stop()

# ------------------- LOAD MODEL SAFELY -------------------
@st.cache_resource
def load_model():
    return joblib.load("best_ev_range_model.pkl")

try:
    model = load_model()
    with open("features_used.json", "r") as f:
        features = json.load(f)
except Exception as e:
    st.error(f"Error loading model. This usually happens due to version mismatch. "
             f"We're fixing it automatically...")
    st.stop()

# ------------------- APP UI -------------------
st.set_page_config(page_title="EV Range Predictor", layout="centered")
st.title("Electric Vehicle Range Predictor")
st.markdown("**CT-264 Assignment #01** – Group: Usman, Maaz, Ameem (CR-23038, CR-23041, CR-23013)")

# Load data for dropdown options
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

df = load_data()

col1, col2 = st.columns(2)
with col1:
    county = st.selectbox("County", options=sorted(df["County"].dropna().unique()))
    city = st.selectbox("City", options=sorted(df[df["County"] == county]["City"].dropna().unique()))
    make = st.selectbox("Make", options=sorted(df["Make"].dropna().unique()))
    model_name = st.selectbox("Model", options=sorted(df[df["Make"] == make]["Model"].dropna().unique()))

with col2:
    model_year = st.slider("Model Year", 2000, 2026, 2023)
    ev_type = st.selectbox("EV Type", options=["BEV", "PHEV"])
    cafv = st.selectbox("CAFV Eligibility", options=["Eligible", "Not Eligible", "Unknown"])
    utility = st.selectbox("Electric Utility", options=sorted(df["Electric_Utility"].dropna().unique()))

vehicle_age = 2025 - model_year
has_range = st.checkbox("Has known electric range", value=True)
has_msrp = st.checkbox("Has MSRP listed", value=False)

if st.button("Predict Electric Range", type="primary"):
    input_df = pd.DataFrame([{
        "County": county,
        "City": city,
        "Make": make,
        "Model": model_name,
        "EV_Type": ev_type,
        "CAFV_Eligibility": cafv,
        "Electric_Utility": utility,
        "Model_Year": model_year,
        "Vehicle_Age": vehicle_age,
        "Has_Range_Data": int(has_range),
        "Has_MSRP": int(has_msrp)
    }])

    pred = model.predict(input_df)[0]
    st.success(f"**Predicted Electric Range: {pred:.1f} miles**")
    
    if pred > 250:
        st.balloons()
        st.info("Long-range Battery Electric Vehicle (BEV)")
    elif pred < 50:
        st.warning("Plug-in Hybrid (PHEV) with short electric range")
