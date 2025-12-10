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
