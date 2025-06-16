import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/sample_uti_data.csv")
    df["symptom_score"] = df[["fever", "pain", "frequency", "nausea", "burning"]].sum(axis=1)
    df["age"] = df["age"].abs() * 20  # Rescale age
    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 60, 100], labels=["child", "adult", "senior"])
    df = pd.get_dummies(df, columns=["age_group"], drop_first=True)
    X = df.drop("uti_diagnosis", axis=1)
    y = df["uti_diagnosis"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
