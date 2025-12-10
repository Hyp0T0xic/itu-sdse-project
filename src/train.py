from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import pandas as pd
import numpy as np

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains an XGBoost Random Forest Classifier using RandomizedSearchCV.
    """
    print("Training model...")
    
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
        verbose=1, # Reduced verbosity for script
        n_iter=10, 
        cv=10
    )
    
    model_grid.fit(X_train, y_train)
    
    print(f"Best params: {model_grid.best_params_}")
    return model_grid
