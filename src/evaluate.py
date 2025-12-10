from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the model using accuracy, confusion matrix, and classification report.
    Prints the results to stdout.
    
    Parameters:
        model: Trained model with reject functionality.
        X_test: Test features.
        y_test: True labels.
    """
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy test: {acc:.3f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return acc, cm
