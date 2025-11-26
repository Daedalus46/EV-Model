import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="⚡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------- Title & Header -------------------
st.markdown("""
    <h1 style='text-align: center; color: #1E90FF;'>Electric Vehicle Range Predictor</h1>
    <h3 style='text-align: center; color: #666666;'>Real-time range prediction using machine learning</h3>
    <hr style='border: 2px solid #1E90FF; border-radius: 5px;'>
""", unsafe_allow_html=True)

# ------------------- Safety Check -------------------
if not os.path.exists("best_ev_range_model.pkl"):
    st.error("Model file missing. Please upload best_ev_range_model.pkl and reboot.")
    st.stop()

# ------------------- Load Model & Data -------------------
@st.cache_resource
def load_model():
    return joblib.load("best_ev_range_model.pkl", mmap_mode='r')

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

model = load_model()
df = load_data()

# ------------------- Sidebar Inputs -------------------
with st.sidebar:
    st.header("Vehicle Details")
    
    county = st.selectbox("County", options=sorted(df["County"].dropna().unique()))
    city_options = df[df["County"] == county]["City"].dropna().unique()
    city = st.selectbox("City", options=sorted(city_options))
    
    make = st.selectbox("Make", options=sorted(df["Make"].dropna().unique()))
    model_options = df[df["Make"] == make]["Model"].dropna().unique()
    model_name = st.selectbox("Model", options=sorted(model_options))
    
    model_year = st.slider("Model Year", 2000, 2026, 2023)
    vehicle_age = 2025 - model_year
    
    ev_type = st.selectbox("EV Type", options=["BEV", "PHEV"])
    cafv = st.selectbox("CAFV Eligibility", options=["Eligible", "Not Eligible", "Unknown"])
    utility = st.selectbox("Electric Utility", options=sorted(df["Electric_Utility"].dropna().unique()))
    
    col_a, col_b = st.columns(2)
    with col_a:
        has_range = st.checkbox("Known Range", value=True)
    with col_b:
        has_msrp = st.checkbox("Has MSRP", value=False)

# ------------------- Prediction -------------------
if st.button("Predict Electric Range", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
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
    
    prediction = model.predict(input_data)[0]
    
    st.markdown(f"""
        <div style='text-align: center; padding: 30px; background-color: #f0f8ff; border-radius: 15px;'>
            <h1 style='color: #1E90FF; margin:0;'>⚡️ {prediction:.1f} miles</h1>
            <p style='color: #666666; font-size: 1.2em;'>Predicted Electric Range</p>
        </div>
    """, unsafe_allow_html=True)
    
    if prediction > 250:
        st.success("Long-range Battery Electric Vehicle (BEV)")
        st.balloons()
    elif prediction > 80:
        st.info("Mid-to-long range electric vehicle")
    else:
        st.warning("Plug-in Hybrid (PHEV) or short-range model")

# ------------------- Footer -------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; color: #888888;'>
        Powered by Random Forest • R² = 0.9957 • MAE = 1.71 miles • Washington State EV Dataset
    </p>
""", unsafe_allow_html=True)

