from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calculate common classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'classification_report': classification_report(y_true, y_pred)
    }
    return metrics

def calculate_binary_cross_entropy(y_true, y_pred_proba, epsilon=1e-9):
    """Calculate binary cross entropy loss"""
    return -np.mean(y_true * np.log(y_pred_proba + epsilon) +
                   (1 - y_true) * np.log(1 - y_pred_proba + epsilon))
