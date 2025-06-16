from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.title("Confusion Matrix")
    plt.show()

    prec, rec, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(rec, prec)
    plt.title("Precision-Recall Curve")
    plt.show()
def explain_model(model, X_sample):
    # If pipeline: get model inside
    clf = model.named_steps['randomforestclassifier'] if 'randomforestclassifier' in model.named_steps else model
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    shap.summary_plot(shap_values[1], X_sample)  