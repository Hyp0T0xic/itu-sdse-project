import pandas as pd
import numpy as np

def describe_numeric_col(x: pd.Series) -> pd.Series:
    """
    Calculates various descriptive stats for a numeric column.
    
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats.
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x: pd.Series, method: str = "mean") -> pd.Series:
    """
    Imputes missing values in a pandas Series.
    
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        # For categorical data, fill with the mode (most frequent value)
        if len(x.mode()) > 0:
            x = x.fillna(x.mode()[0])
    return x

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    """
    return pd.read_csv(file_path)

def filter_by_date(df: pd.DataFrame, date_col: str, min_date: str, max_date: str) -> pd.DataFrame:
    """
    Filters the dataframe by a date range.
    """
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    min_d = pd.to_datetime(min_date).date()
    max_d = pd.to_datetime(max_date).date()
    return df[(df[date_col] >= min_d) & (df[date_col] <= max_d)]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data cleaning:
    1. Drops unnecessary columns.
    2. Removes rows with empty target variables.
    3. Filters for 'signup' source.
    """
    # Columns to drop based on Feature Selection in notebook
    drop_cols = [
        "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen",
        "domain", "country", "visited_learn_more_before_booking", "visited_faq"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Convert empty strings to NaN
    cols_to_fix = ["lead_indicator", "lead_id", "customer_code"]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan)

    # Drop rows with missing critical info
    df = df.dropna(subset=["lead_indicator", "lead_id"])

    # Filter for source == 'signup'
    if "source" in df.columns:
        df = df[df.source == "signup"]
        
    return df
