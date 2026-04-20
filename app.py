# app.py

import streamlit as st
import joblib
import numpy as np

model = joblib.load("tms_model.pkl")

st.title("TMS Response Predictor")

FastRipples = st.number_input("FastRipples")
Spikes = st.number_input("Spikes")
NREM = st.number_input("NREM")
REM = st.number_input("REM")
Wake = st.number_input("Wake")
DPI = st.number_input("DPI")
TBI = st.selectbox("TBI (0/1)", [0,1])
SD = st.selectbox("SD (0/1)", [0,1])

if st.button("Predict"):
    X = np.array([[TBI, SD, DPI, FastRipples, Spikes, NREM, REM, Wake]])
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    if pred == 1:
        st.success(f"Use TMS ✅ (Confidence: {prob:.2f})")
    else:
        st.error(f"Do NOT use TMS ❌ (Confidence: {1-prob:.2f})")