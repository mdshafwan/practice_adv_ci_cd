"""sample"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix
)

def evalution(y_test,pred):
    """sample"""
    return {
        "accuracy_scare":accuracy_score(y_test,pred),
        "f1_score":f1_score(y_test,pred),
        "recall_Score":recall_score(y_test,pred),
        "precision_score":precision_score(y_test,pred)
    }

def get_confusion_matrix(y_test,pred):
    """sample"""
    return confusion_matrix(y_test,pred)

def get_classification_report(y_test,pred):
    """sample"""
    return classification_report(y_test,pred)
