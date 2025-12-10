import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from src.data import load_data, clean_data, filter_by_date, remove_outliers, impute_cols, engineer_features
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # 1. Load Data
    print("Loading data...")
    # Assuming data is already pulled via DVC to this path
    if not os.path.exists("artifacts/raw_data.csv"):
        print("Error: artifacts/raw_data.csv not found. Did you run 'dvc pull'?")
        return

    df = load_data("artifacts/raw_data.csv")
    
    # 2. Basic Cleaning & Filtering
    print("Cleaning data...")
    df = clean_data(df)
    
    # Using specific dates from notebook logic
    df = filter_by_date(df, "date_part", "2024-01-01", "2024-01-31")
    
    # 3. Feature Engineering & Preprocessing
    print("Feature Engineering...")
    df = impute_cols(df)
    df = engineer_features(df)
    
    # Identify continuous columns for outlier removal
    # We select standard numeric types
    cont_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df = remove_outliers(df, cont_cols)
    
    # 4. Encoding & Scaling
    # Define categorical columns we want to encode (based on notebook)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    # Only use the ones that are actually present
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    print("Encoding categorical variables...")
    # pandas get_dummies is simpler than the manual helper function
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
    
    # 5. Prepare X and y
    if "lead_indicator" not in df.columns:
        raise ValueError("Target column 'lead_indicator' not found in dataset!")
        
    y = df["lead_indicator"]
    # Drop target and non-feature columns
    X = df.drop(columns=["lead_indicator", "date_part", "lead_id", "customer_code"], errors='ignore')
    
    # Scale X
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Save Scaler for later inference
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    
    # 6. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, random_state=42, test_size=0.15, stratify=y
    )
    
    # 7. Train
    print("Starting training...")
    model_grid = train_model(X_train, y_train)
    
    # 8. Evaluate
    evaluate_model(model_grid, X_test, y_test)
    
    # 9. Save Model
    print("Saving best model...")
    joblib.dump(model_grid.best_estimator_, "artifacts/model.joblib")
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
