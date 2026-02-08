import numpy as np
import pandas as pd

file_path = "C:\Users\Spike\OneDrive - National University of Singapore\Desktop\NUS\Improving my Coding\Kaggle Competitions\Akkadian to English (Data)\data\bibliography.csv"


def load_bibliography_data(file_path):
    """
    Load bibliography data from a CSV file.

    Parameters:
    """
    file_path (str): The path to the CSV file containing bibliography data.

    Returns:
    pd.DataFrame: A DataFrame containing the bibliography data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()
