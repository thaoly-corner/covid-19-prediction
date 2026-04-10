import pandas as pd
from modules.config import CATEGORICAL_FEATURES, DATE_COLUMNS
from modules.utils import standardize_dates
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_data(df):
    """
    Main cleaning pipeline that handles duplicates, feature engineering,
    and missing values.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Feature Engineering: Create 'is_dead' target and drop unnecessary columns
    df = standardize_dates(df, DATE_COLUMNS)

    if 'date_died' in df.columns:
        df['is_dead'] = df['date_died'].apply(lambda x: 0 if x == '9999-99-99' else 1)

    df['entry_date'] = pd.to_datetime(df['entry_date'], dayfirst=True, errors='coerce')
    df['date_symptoms'] = pd.to_datetime(df['date_symptoms'], dayfirst=True, errors='coerce')

    df['delay_time'] = (df['entry_date'] - df['date_symptoms']).dt.days
    df.loc[df['delay_time'] < 0, 'delay_time'] = 0
    
    # Drop original date columns and ID columns
    cols_to_drop = ['date_died', 'id', 'entry_date', 'date_symptoms']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 3. Standardize Missing Codes (98, 99 -> 99)
    existing_cats = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    df[existing_cats] = df[existing_cats].replace([98, 99], 99)
    
    # 4. Standardize Yes/No: Convert 1/2 (Yes/No) to 1/0
    # Exclude columns that are already formatted or are numerical
    cols_to_map = [c for c in existing_cats if c not in ['covid_res']]
    df[cols_to_map] = df[cols_to_map].replace({1: 1, 2: 0})
    return df

def convert_cat(df):
    """Converts identified features to Categorical type."""
    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns:
            df[feature] = df[feature].astype('category')
    return df


def apply_feature_binning(df, strategy='ordinal'):
    """
    Apply custom binning for 'age' and 'delay_time'.
    
    Args:
        df (pd.DataFrame): The dataframe to process.
        strategy (str): 'ordinal' for numeric labels (0, 1, 2...), 
                        'label' for string names ('Children', 'Seniors'...).
    Returns:
        pd.DataFrame: Dataframe with binned features.
    """
    df = df.copy()
    
    # 1. Define Age Bins: 0-12, 13-19, 20-39, 40-59, 60+
    # pd.cut works as (left, right], so we set edges carefully
    age_bins = [-1, 12, 19, 39, 59, np.inf]
    age_labels = ['Child', 'Teenager', 'Young Adult', 'Middle-aged', 'Senior']
    
    # 2. Define Delay Bins: 0, 1-7, 8-10, 11-15, 15+
    # Adjusting based on your snippet logic
    delay_bins = [-np.inf, 0, 7, 10, 15, np.inf]
    delay_labels = ['Short/No Delay', 'Medium', 'Long', 'Very Long', 'Extreme']

    if strategy == 'ordinal':
        # labels=False returns 0, 1, 2, 3...
        # df['age_ordinal'] = pd.cut(df['age'], bins=age_bins, labels=False)
        df['delay_ordinal'] = pd.cut(df['delay_time'], bins=delay_bins, labels=False)
    else:
        # returns string labels
        # df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        df['delay_group'] = pd.cut(df['delay_time'], bins=delay_bins, labels=delay_labels)

    # Handle any potential NaNs from out-of-range values
    # cols_to_fix = [c for c in ['age_ordinal', 'delay_ordinal'] if c in df.columns]
    cols_to_fix = [c for c in ['delay_ordinal'] if c in df.columns]
    df[cols_to_fix] = df[cols_to_fix].fillna(-1).astype(int)
    # df.drop(columns=['age', 'delay_time'], inplace=True, errors='ignore')
    df.drop(columns=['delay_time'], inplace=True, errors='ignore')
    
    return df

def get_preprocessor(cat_cols=None, ord_cols=None, num_cols=None):
    """
    Builds a Scikit-Learn ColumnTransformer for data preprocessing.
    
    Args:
        cat_cols (list): List of nominal categorical columns (need One-Hot Encoding).
        ord_cols (list): List of ordinal columns (already integer encoded, just passthrough).
        num_cols (list): List of continuous numerical columns (need Standardization).
        
    Returns:
        ColumnTransformer: The compiled preprocessor ready for fit_transform.
    """
    transformers = []

    # 1. Nominal Categorical Features -> One-Hot Encoding
    # drop='first' is crucial for Logistic Regression to avoid the Dummy Variable Trap
    if cat_cols and len(cat_cols) > 0:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
        )

    # 2. Ordinal Features -> Passthrough
    # Since you already binned them into 0, 1, 2, 3 in the cleaning step
    if ord_cols and len(ord_cols) > 0:
        transformers.append(
            ('ord', 'passthrough', ord_cols)
        )

    # 3. Numerical Features -> Standard Scaling (Mean=0, Std=1)
    if num_cols and len(num_cols) > 0:
        transformers.append(
            ('num', StandardScaler(), num_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    return preprocessor