import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from src.data import load_data, clean_data, filter_by_date, remove_outliers, impute_cols, engineer_features
from src.train import train_xgboost, train_log_reg, select_best_model
from src.evaluate import evaluate_model
import mlflow

# Auto-log all sklearn and xgboost params, metrics, and models
mlflow.sklearn.autolog()
mlflow.xgboost.autolog()

EXPERIMENT_NAME = "Lead_Scoring_Production"

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        print(f"Starting MLflow run: {run.info.run_id}")
        
        # 1. Load Data
        print("Loading data...")
        if not os.path.exists("artifacts/raw_data.csv"):
            print("Error: artifacts/raw_data.csv not found. Did you run 'dvc pull'?")
            return

        df = load_data("artifacts/raw_data.csv")
    
        # 2. Basic Cleaning & Filtering
        print("Cleaning data...")
        df = clean_data(df)
        df = filter_by_date(df, "date_part", "2024-01-01", "2024-01-31")
        
        # 3. Feature Engineering & Preprocessing
        print("Feature Engineering...")
        df = impute_cols(df)
        df = engineer_features(df)
        
        # Identify continuous columns for outlier removal
        cont_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        df = remove_outliers(df, cont_cols)
        
        # 4. Encoding & Scaling
        cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        
        print("Encoding categorical variables...")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
        
        # 5. Prepare X and y
        if "lead_indicator" not in df.columns:
            raise ValueError("Target column 'lead_indicator' not found in dataset!")
            
        y = df["lead_indicator"]
        X = df.drop(columns=["lead_indicator", "date_part", "lead_id", "customer_code"], errors='ignore')
        
        # Scale X
        print("Scaling features...")
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler, "artifacts/scaler.pkl")
        
        # 6. Split Data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, random_state=42, test_size=0.15, stratify=y
        )
        
        # 7. Train & Select
        print("Starting training phase...")
        model_xgb_grid = train_xgboost(X_train, y_train)
        model_lr_grid = train_log_reg(X_train, y_train)
        
        # Select best model based on F1 Score
        best_model = select_best_model(model_xgb_grid, model_lr_grid, X_test, y_test)
        mlflow.set_tag("winner", type(best_model).__name__)
        
        # 8. Evaluate Best Model
        evaluate_model(best_model, X_test, y_test)
        
        # 9. Save Best Model
        print("Saving best model...")
        joblib.dump(best_model, "artifacts/model.pkl")
        print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
