import pandas as pd
import kagglehub
from ultralytics import YOLO
from disease_recognition.params import *

def load_data(dataset: str="lokisilvres/dental-disease-panoramic-detection-dataset") -> str:

    path = kagglehub.dataset_download("lokisilvres/dental-disease-panoramic-detection-dataset")

    return path
