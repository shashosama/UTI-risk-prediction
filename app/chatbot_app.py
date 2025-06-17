import streamlit as st
import pandas as pd
import joblib

st.title("UTI Risk Chatbot")

# Load model
model = joblib.load("model/uti_model.pkl")

user_input = st.text_input("Describe your symptoms:")

def extract_features(text):
    text = text.lower()
    features = {
        "fever": int("fever" in text),
        "burning": int("burning" in text),
        "pain": int("pain" in text or "ache" in text),
        "urgency": int("urgency" in text or "frequent urination" in text),
        # Add more features
    }
    return features

if user_input.strip():
    features = extract_features(user_input)
    df = pd.DataFrame([features])

    prob = model.predict_proba(df)[0, 1]

    st.write(f"Predicted UTI risk: {prob:.2%}")
    if prob > 0.5:
        st.warning("High risk of UTI. Please consult a doctor.")
    else:
        st.success(" Low risk of UTI.")
else:
    st.info("Please describe your symptoms above.")
