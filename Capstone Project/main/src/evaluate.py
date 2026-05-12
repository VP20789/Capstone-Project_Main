from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate evaluation metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    """

    return confusion_matrix(y_true, y_pred)


def generate_classification_report(y_true, y_pred):
    """
    Generate classification report.
    """

    return classification_report(y_true, y_pred)