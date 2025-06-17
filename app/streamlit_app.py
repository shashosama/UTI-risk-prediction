import shap
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model/uti_model.pkl")  # Adjust path to your model

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

    # SHAP for individual input
    explainer = shap.Explainer(model.named_steps["randomforestclassifier"])
    shap_values = explainer(df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.waterfall_plot(shap_values[0], show=False))

uploaded_file = st.file_uploader("Upload CSV")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    prob = model.predict_proba(input_df)[:, 1]
    input_df['UTI Risk'] = prob
    st.dataframe(input_df)

    # SHAP explainability
    explainer = shap.Explainer(model.named_steps["randomforestclassifier"])
    shap_values = explainer(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.summary_plot(shap_values, input_df, show=False))

    # Download button (in-memory)
    csv = input_df.to_csv(index=False)
    st.download_button("Download Predictions", csv, file_name="predictions.csv", mime="text/csv")
