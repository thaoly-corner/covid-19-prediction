import kagglehub
import pandas as pd
import os

def download_and_load(handle, filename):
    """
    Downloads the dataset via kagglehub and loads the specific CSV file.

    Args:
        handle (str): Kaggle dataset handle (username/dataset).
        filename (str): Name of the CSV file to load.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    # Downloads the latest version of the dataset
    path = kagglehub.dataset_download(handle)
    
    # Locate the CSV file within the downloaded path
    csv_path = os.path.join(path, filename)
    
    return pd.read_csv(csv_path)