from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

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
