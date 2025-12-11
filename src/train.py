from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from scipy.stats import uniform, randint
import pandas as pd
import numpy as np

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains an XGBoost Random Forest Classifier using RandomizedSearchCV.
    """
    print("Training XGBoost...")
    
    model = XGBRFClassifier(random_state=42)
    
    # Hyperparameters from the notebook
    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }
    
    # Grid search
    model_grid = RandomizedSearchCV(
        model, 
        param_distributions=params, 
        n_jobs=-1, 
        verbose=1, 
        n_iter=10, 
        cv=10
    )
    
    model_grid.fit(X_train, y_train)
    print(f"XGBoost Best params: {model_grid.best_params_}")
    return model_grid

def train_log_reg(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains a Logistic Regression model using RandomizedSearchCV.
    """
    print("Training Logistic Regression...")
    
    model = LogisticRegression()
    
    # Hyperparameters from the notebook (cell 853)
    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["l1", "l2", "elasticnet"], # Removed "none" as it can cause issues with some solvers/versions
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    
    model_grid = RandomizedSearchCV(
        model, 
        param_distributions=params, 
        verbose=1, 
        n_iter=10, 
        cv=3,
        n_jobs=-1
    )
    
    model_grid.fit(X_train, y_train)
    print(f"LogReg Best params: {model_grid.best_params_}")
    return model_grid

def select_best_model(model_xgb, model_lr, X_test, y_test):
    """
    Compares two models based on F1 Score on the test set.
    Returns the best best estimator.
    """
    print("Selecting best model...")
    
    # Predict
    y_pred_xgb = model_xgb.predict(X_test)
    y_pred_lr = model_lr.predict(X_test)
    
    # Score
    f1_xgb = f1_score(y_test, y_pred_xgb)
    f1_lr = f1_score(y_test, y_pred_lr)
    
    print(f"XGBoost F1 Score: {f1_xgb:.4f}")
    print(f"LogReg F1 Score:  {f1_lr:.4f}")
    
    if f1_xgb >= f1_lr:
        print(">>> Winner: XGBoost")
        return model_xgb.best_estimator_
    else:
        print(">>> Winner: Logistic Regression")
        return model_lr.best_estimator_
