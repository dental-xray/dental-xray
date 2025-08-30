import kagglehub
from disease_recognition.params import *

def load_data(dataset):

    """Load dataset from Kaggle Hub"""
    print("=== Loading dataset from Kaggle Hub ===")
    print(f"Dataset: {dataset}")

    path = kagglehub.dataset_download(dataset)

    print(f"✅ Dataset downloaded to: {path}")
    print()

    return path
