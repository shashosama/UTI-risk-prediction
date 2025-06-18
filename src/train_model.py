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
    print("Model trained and saved to model/uti_model.pkl")

if __name__ == "__main__":
    df = pd.read_csv("data/uti_real_data.csv")

    # Clean target
    df["Inflammation of urinary bladder"] = (
        df["Inflammation of urinary bladder"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"yes": 1, "no": 0})
    )
    df["Inflammation of urinary bladder"] = pd.to_numeric(df["Inflammation of urinary bladder"], errors="coerce")
    df = df.dropna(subset=["Inflammation of urinary bladder"])

    # Define bool_cols
    bool_cols = [
        "Occurrence of nausea",
        "Lumbar pain",
        "Urine pushing (continuous need for urination)",
        "Micturition pains",
        "Burning of urethra, itch, swelling of urethra outlet"
    ]

    # Clean symptom columns
    for col in bool_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"yes": 1, "no": 0, "true": 1, "false": 0})
        )

    # Convert all to numeric at once
    df[bool_cols] = df[bool_cols].apply(pd.to_numeric, errors="coerce")
    print("Any NaNs in bool_cols:\n", df[bool_cols].isna().sum())

    # Drop rows where symptom columns couldn't convert
    df = df.dropna(subset=bool_cols)

    # Create symptom_score
    df["symptom_score"] = df[bool_cols].sum(axis=1)

    # Create X and y
    X = df[["Temperature of patient", "symptom_score"]]
    y = df["Inflammation of urinary bladder"].astype(int)

    # Debug print to see final shape
    print(f"Final X shape: {X.shape}, Final y length: {len(y)}")

    # Split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train + save
    train_and_save(X_train, y_train)
