import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.joblib")

st.title("UTI Risk Predictor")
age = st.slider("Age", 0, 100)
symptom_score = st.slider("Symptom Score (0-5)", 0, 5)
age_group = st.selectbox("Age Group", ["adult", "senior"])
data = {
    "age": [age],
    "symptom_score": [symptom_score],
    "age_group_senior": [1 if age_group == "senior" else 0]
}

df = pd.DataFrame(data)
if st.button("Predict"):
    prob = model.predict_proba(df)[0][1]
    st.success(f"Predicted UTI Risk: {prob:.2%}")
