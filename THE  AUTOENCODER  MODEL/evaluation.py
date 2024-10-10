from sklearn.metrics import classification_report, confusion_matrix

def evaluate_performance(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
