from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def train_and_save(X_train, y_train):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(class_weight="balanced", random_state=42)
    )

    model.fit(X_res, y_res)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/uti_model.pkl")
    print(" Model trained and saved to model/uti_model.pkl")

if __name__ == "__main__":
    # Example: load from real or synthetic data
    df = pd.read_csv("data/uti_real_data.csv")
    print("Columns:", df.columns.tolist())
    print(df.head())
    print("Unique target values before replace:", df["Nephritis of renal pelvis origin"].unique())

    df = df.replace({'yes': 1, 'no': 0})

    X = df.drop("Nephritis of renal pelvis origin", axis=1)
    y = df["Nephritis of renal pelvis origin"].map({"no": 0, "yes": 1})
    print("Any NaNs in y?", df["Nephritis of renal pelvis origin"].isna().sum())
    df = df.dropna(subset=["Nephritis of renal pelvis origin"])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_save(X_train, y_train)
