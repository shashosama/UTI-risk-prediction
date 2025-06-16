import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/uti_real_data.csv"):
    df = pd.read_csv(path)
    
    # Example: clean and prepare
    # Ensure temperature is numeric
    df["Temperature of patient"] = pd.to_numeric(df["Temperature of patient"], errors="coerce")
    
    # Example symptom score: sum of key boolean features (convert True/False to int)
    bool_cols = [
        "Occurrence of nausea",
        "Lumbar pain",
        "Urine pushing (continuous need for urination)",
        "Micturition pains",
        "Burning of urethra, itch, swelling of urethra outlet"
    ]
    
    for col in bool_cols:
        df[col] = df[col].astype(int)

    df["symptom_score"] = df[bool_cols].sum(axis=1)
    
    # Create feature matrix
    X = df[
        ["Temperature of patient", "symptom_score"]
    ]
    
    # Example target: inflammation of urinary bladder
    y = df["Inflammation of urinary bladder"].astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42)
