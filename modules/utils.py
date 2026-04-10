import seaborn as sns
import matplotlib.pyplot as plt

def display_unique_values(df):
    """
    Iterates through all columns in the DataFrame and prints their unique values.
    Useful for identifying missing codes (97, 98, 99) or data inconsistencies.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    """
    print("\n" + "="*50)
    print("SUMMARY: UNIQUE VALUES PER COLUMN")
    print("="*50)
    
    for column in df.columns:
        unique_vals = df[column].unique()
        print(f"Feature: {column}")
        print(f"Values: {unique_vals}")
        print("-" * 30)
    
    print("Inspection Complete.\n")

def standardize_dates(df, date_cols):
    """
    Standardizes multiple date columns to YYYY-MM-DD format.
    Handles '9999-99-99' as a string to avoid datetime errors.
    """
    for col in date_cols:
        if col in df.columns:
            # 1. Clean whitespace and replace '/' with '-'
            df[col] = df[col].astype(str).str.strip().str.replace('/', '-')
    return df
