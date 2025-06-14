from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=1000, n_features=6, weights=[0.7, 0.3],
                           n_informative=4, n_redundant=1, random_state=42)
df = pd.DataFrame(X, columns=["age", "fever", "pain", "frequency", "nausea", "burning"])
df["uti_diagnosis"] = y
df.to_csv("data/sample_uti_data.csv", index=False)
