import streamlit as st
import pandas as pd
import joblib

st.title("UTI Risk Predictor")

model = joblib.load("model/uti_model.pkl")

# Input for features model expects
temperature = st.slider("Temperature of patient", 35.0, 41.0, 37.0)
symptom_score = st.slider("Symptom Score (0-5)", 0, 5, 2)

# Build DataFrame with matching features
df = pd.DataFrame([[temperature, symptom_score]], columns=["Temperature of patient", "symptom_score"])

if st.button("Predict"):
    prob = model.predict_proba(df)[0,1]
    st.write(f"Predicted UTI risk: {prob:.2%}")
    if prob > 0.5:
        st.warning(" High risk of UTI. Please consult a doctor.")
    else:
        st.success("Low risk of UTI.")
