import kagglehub
from disease_recognition.params import *

def load_data(dataset) -> str:

    path = kagglehub.dataset_download(dataset)

    return path
