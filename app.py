# app.py
import streamlit as st
import pandas as pd
import joblib
import json

# Load model and features
model = joblib.load('best_ev_range_model.pkl')
with open('features_used.json') as f:
    features = json.load(f)

# Title
st.set_page_config(page_title="EV Range Predictor", layout="centered")
st.title("Electric Vehicle Range Predictor")
st.markdown("### Built using Washington State EV Data (CT-264 Assignment)")

# Load original data to get unique values for dropdowns
@st.cache_data
def load_options():
    df = pd.read_csv("cleaned_data.csv")
    options = {}
    for col in ['County', 'City', 'Make', 'Model', 'EV_Type', 'CAFV_Eligibility', 'Electric_Utility']:
        options[col] = sorted(df[col].dropna().unique().tolist())
    return options

opts = load_options()

# Input fields
col1, col2 = st.columns(2)

with col1:
    county = st.selectbox("County", options=opts['County'])
    city = st.selectbox("City", options=opts['City'])
    make = st.selectbox("Make", options=opts['Make'])
    model_name = st.selectbox("Model", options=opts['Model'])

with col2:
    model_year = st.number_input("Model Year", 2000, 2030, 2023)
    ev_type = st.selectbox("EV Type", options=opts['EV_Type'])
    cafv = st.selectbox("CAFV Eligibility", options=opts['CAFV_Eligibility'])
    utility = st.selectbox("Electric Utility", options=opts['Electric_Utility'])

vehicle_age = 2025 - model_year
has_range_data = st.checkbox("Has Range Data (known range)", value=True)
has_msrp = st.checkbox("Has MSRP Listed", value=False)

if st.button("Predict Electric Range", type="primary"):
    input_data = pd.DataFrame([{
        'County': county,
        'City': city,
        'Make': make,
        'Model': model_name,
        'EV_Type': ev_type,
        'CAFV_Eligibility': cafv,
        'Electric_Utility': utility,
        'Model_Year': model_year,
        'Vehicle_Age': vehicle_age,
        'Has_Range_Data': int(has_range_data),
        'Has_MSRP': int(has_msrp)
    }])

    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Electric Range: **{prediction:.1f} miles**")
    
    if prediction > 250:
        st.balloons()
        st.markdown("Long-range BEV detected!")
    elif prediction < 50:
        st.warning("This is likely a PHEV with short electric range.")