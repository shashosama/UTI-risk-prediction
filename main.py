from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import yaml
import mlflow

# Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=6, weights=[0.7, 0.3],
                           n_informative=4, n_redundant=1, random_state=42)
df = pd.DataFrame(X, columns=["age", "fever", "pain", "frequency", "nausea", "burning"])
df["uti_diagnosis"] = y
df.to_csv("data/sample_uti_data.csv", index=False)

def compare_models(X, y):
    models = {
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'SVM': SVC(probability=True, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=(y==0).sum() / (y==1).sum())
    }

    mlflow.start_run()

    for name, model in models.items():
        grid = GridSearchCV(model, param_grid={}, cv=3, scoring='roc_auc')
        grid.fit(X, y)
        best_auc = grid.best_score_
        print(f"{name} ROC AUC: {best_auc:.3f}")

        # Log to MLflow
        mlflow.log_metric(f"{name}_roc_auc", best_auc)

    mlflow.end_run()

def run_chatbot():
    print("Chatbot functionality placeholder — implement your Streamlit app here.")

if __name__ == "__main__":
    compare_models(X, y)
    run_chatbot()
