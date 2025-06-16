from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def train_and_save(X_train, y_train):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = make_pipeline(StandardScaler(), RandomForestClassifier(class_weight="balanced"))
    model.fit(X_res, y_res)
    joblib.dump(model, "model.joblib")
