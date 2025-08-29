import pandas as pd
from ultralytics import YOLO
from disease_recognition.params import *

def save_results(params: dict, metrics: dict) -> None:
    pass

def save_model(model: ultralytics.models.yolo.model.YOLO) -> None:
    pass

def load_model(stage="Production") -> ultralytics.models.yolo.model.YOLO:
    pass

def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    pass

def mlflow_run(func):
    pass
