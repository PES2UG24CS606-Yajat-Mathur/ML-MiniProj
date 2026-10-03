import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    average_precision_score, 
    precision_recall_curve
)

def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluates model performance using AUPRC, AUROC, Confusion Matrix, and Recall.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    
    print(f"\n================ {model_name} ================")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_cm.png")
    plt.close()
    
    return {"Model": model_name, "AUROC": auroc, "AUPRC": auprc}