import os
import sys

# from api import predict
from api.predict import predict
from api.predict import PredictionResult
from api.load import load_model
from disease_recognition.params import *


# Load the data and model at startup

# google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# if google_credentials_path is None:
#     print("❌ GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
# elif not os.path.exists(google_credentials_path):
#     print(f"❌ Google credentials file not found at {google_credentials_path}")

model = load_model(
    model_storage="local",
    stage="None",
    bucket_name=BUCKET_NAME,
    path=LOCAL_REGISTRY_PATH,
    filename="default.pt",
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    mlflow_model_name=MLFLOW_MODEL_NAME
)

def run_prediction(image_path: str) -> PredictionResult:
    """
    Runs prediction on a given image path.
    Returns a list of dicts, each containing:
        - class: int
        - confidence: float
        - bbox: [x1, y1, x2, y2]
    """
    # Ensure full path
    full_path = os.path.join(image_path) if not os.path.isabs(image_path) else image_path

    # Run YOLO prediction
    return predict(model, full_path)


if __name__ == "__main__":
    # For testing without FastAPI
    pass
