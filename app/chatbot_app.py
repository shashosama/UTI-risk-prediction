import streamlit as st
import pandas as pd
import joblib

st.title("UTI Risk Chatbot")

model = joblib.load("model/uti_model.pkl")

user_input = st.text_input("Describe your symptoms:")

def extract_symptom_score(text):
    symptoms = ["fever", "burning", "pain", "urgency"]
    score = sum(1 for s in symptoms if s in text.lower())
    return score

if user_input:
    symptom_score = extract_symptom_score(user_input)
    
    # For now, use a default temperature — or let user type it
    temp = st.number_input("Enter your temperature (°C)", 35.0, 41.0, 37.0)

    # Build DataFrame with the correct columns
    df = pd.DataFrame([[temp, symptom_score]], columns=["Temperature of patient", "symptom_score"])
    
    # Predict
    prob = model.predict_proba(df)[0, 1]
    
    st.write(f"Predicted UTI risk: {prob:.2%}")
    if prob > 0.5:
        st.warning(" High risk of UTI. Please consult a doctor.")
    else:
        st.success(" Low risk of UTI.")
