from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pandas as pd
import numpy as np
import mlflow

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
    
    # Log metrics to MLflow if a run is active
    if mlflow.active_run():
        mlflow.log_metric("test_accuracy", acc)
        f1 = f1_score(y_test, y_pred, average='weighted') # Calculate F1 as well since it's important
        mlflow.log_metric("test_f1_score", f1)
        print(f"Logged metrics to MLflow run: {mlflow.active_run().info.run_id}")

    return acc, cm
