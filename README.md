# UTI Risk Prediction  


This project builds a **machine learning model** to predict the risk of **urinary tract infection (UTI)** based on patient symptoms and temperature.  
It provides two interactive app interfaces:
-  A standard **form-based Streamlit app**
-  A **chatbot-style Streamlit app**  

The goal is to support early detection of UTI-related conditions, such as:
- **Inflammation of the urinary bladder** (main target)
- **Nephritis of renal pelvis origin** (optional target for future extension)

---
# General Problem
# UTIs are common and can cause serious issues (e.g., inflammation of urinary bladder, nephritis of renal pelvis origin)
# Symptoms are subjective, vary across patients, and fast, data-driven assessments are lacking.

# Solution: Machine learning model in R to predict UTI risk based on symptoms and temperature.

##  Dataset  

The dataset contains **120 patient records** with features:
- `Temperature of patient` (numeric)  
- `Occurrence of nausea` (yes/no)  
- `Lumbar pain` (yes/no)  
- `Urine pushing (continuous need for urination)` (yes/no)  
- `Micturition pains` (yes/no)  
- `Burning of urethra, itch, swelling of urethra outlet` (yes/no)  

Targets:
- `Inflammation of urinary bladder` (yes/no)
- `Nephritis of renal pelvis origin` (yes/no)

---

##  Features  

 **Data processing**
- Converts yes/no symptom columns to numeric (0/1)
- Creates `symptom_score` (sum of key symptom indicators)

 **Modeling**
- Random Forest Classifier (with SMOTE for class balancing)
- Trained on `Temperature of patient` + `symptom_score`
- Metrics: confusion matrix, ROC-AUC, precision-recall

 **Apps**
- **Form App:** sliders for temperature + symptom score, displays risk level
- **Chatbot App:** accepts typed symptom descriptions, extracts features, predicts risk

**Planned Extensions**
- SHAP explainability  
- Visualizations (ROC curves, feature importance)  
- Support for additional features (e.g., age, age group)

---

##  How to Run  

### Locally  

Run the form app:  
```bash
streamlit run app/streamlit_app.py
streamlit run app/chatbot_app.py

#  What I have learned:
#  Data preprocessing: cleaning categorical variables, creating symptom_score
#  Model building: using caret for classification (RF, logistic regression, etc.)
#  Performance evaluation: confusion matrix, ROC-AUC
# Visualization: ROC curve with ggplot2 / plotly
#  Reproducible ML workflow: clear data split, model train, eval, plot