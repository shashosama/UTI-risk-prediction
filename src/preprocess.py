import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath=None):
    if filepath is None:
        # Build absolute path from project root dynamically
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "uti_real_data.csv"))
    
    print("Loading data from:", filepath)  # Debug print
    df = pd.read_csv(filepath)

    # Convert yes/no to 1/0
    bool_cols = [
    'Occurrence of nausea',
    'Lumbar pain',
    'Urine pushing (continuous need for urination)',
    'Micturition pains',
    'Burning of urethra, itch, swelling of urethra outlet'
    ]

    for col in bool_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df['Inflammation of urinary bladder'] = df['Inflammation of urinary bladder'].map({'yes': 1, 'no': 0})
    df['symptom_score'] = df[bool_cols].sum(axis=1)

    X = df.drop(columns=['Inflammation of urinary bladder'])
    y = df['Inflammation of urinary bladder']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
