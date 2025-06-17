import streamlit as st
import pandas as pd
import joblib

st.title("UTI Risk Chatbot")

# Load model once
model = joblib.load("model/uti_model.pkl")

# Input box
user_input = st.text_input("Describe your symptoms:")

def extract_features(text):
    features = {
        "fever": int("fever" in text.lower()),
        "burning": int("burning" in text.lower()),
        "pain": int("pain" in text.lower()),
        "urgency": int("urgency" in text.lower()),
        # Add more as needed
    }
    return features

if user_input:
    features = extract_features(user_input)
    df = pd.DataFrame([features])

    prob = model.predict_proba(df)[0, 1]

    st.write(f"Predicted UTI risk: {prob:.2%}")
    if prob > 0.5:
        st.warning(" High risk of UTI. Please consult a doctor.")
    else:
        st.success("Low risk of UTI.")
