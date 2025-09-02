import kagglehub
from disease_recognition.params import *

"""Data loading functions for disease recognition project"""

def load_data(dataset: str) -> str:

    """Load dataset from Kaggle Hub

    Args:
        dataset (str): The Kaggle dataset identifier in the format "username/dataset-name"
    Returns:
        str: The local path where the dataset is downloaded
    Raises:
        ValueError: If the dataset identifier is invalid or the download fails
    Example:
        path = load_data("username/dataset-name")
    """

    print("=== Loading dataset from Kaggle Hub ===")
    print(f"Dataset: {dataset}")

    path = kagglehub.dataset_download(dataset)

    print(f"✅ Dataset downloaded to: {path}")
    print()

    return path
