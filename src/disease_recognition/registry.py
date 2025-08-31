import pandas as pd
from ultralytics import YOLO
from disease_recognition.params import *
from datetime import datetime
from PIL import Image
import os
from google.cloud import storage
# from tensorflow import keras
import mlflow
from mlflow.tracking import MlflowClient

def save_results(params: dict, metrics: dict):

    """Save training/validation results to a CSV file"""
    print("=== Saving results ===")
    print(f"params: {params}")
    print(f"metrics: {metrics}")

    if MODEL_TARGET == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on mlflow")


def save_model(model, model_storage, path, filename=None):

    """Save the trained model to the specified model_storage"""

    print("=== Saving model ===")
    print(f"model_storage: {model_storage}")
    print(f"path: {path}")
    print(f"filename: {filename}")

    if filename is None:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'trained_model_{current_time}.pt'

    model_path = os.path.join(path, filename)
    model.save(model_path)

    print("✅ Model saved locally")

    if model_storage == "local":
        pass

    elif model_storage == "gcs":

        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    elif model_storage == "mlflow":

        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )

        print("✅ Model saved to MLflow")

    else:
        print("❌ Model target not recognized. Please choose either 'local', 'gcs' or 'mlflow'.\n")

    print()

    return None


def load_model(model_storage, stage="Production",bucket_name=None, filename=None, path=None):

    """Load a model from the specified model_storage"""

    print("=== Loading model ===")
    print(f"model_storage: {model_storage}")
    print(f"stage: {stage}")
    print(f"filename: {filename}")
    print(f"path: {path}")

    if model_storage == "local":
        model = YOLO(os.path.join(path, filename))

        print("✅ Model loaded from local file")

    elif model_storage == "gcs":

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix="models/"))

        if len(blobs) == 0:
            raise ValueError("No model found in GCS bucket")

        if filename is not None:
            blob_name = f"models/{filename}"

            blob = bucket.blob(blob_name)
        else:
            blob = max(blobs, key=lambda b: b.updated)
            filename = blob.name.split("/")[-1]

        local_path = os.path.join(path, filename)
        blob.download_to_filename(local_path)

        model = YOLO(local_path)
        print("✅ Model loaded from GCS")

    elif model_storage == "mlflow":
        # # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        # import mlflow
        # from mlflow.tracking import MlflowClient

        # client = MlflowClient()
        # latest_model_version = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=[stage])[0]
        # model_uri = f"models:/{MLFLOW_MODEL_NAME}/{latest_model_version.version}"
        # model = mlflow.pyfunc.load_model(model_uri)

        print("✅ Model loaded from MLflow")

    else:
        print("❌ Model target not recognized. Please choose either 'local', 'gcs' or 'mlflow'.")

    print()
    return model


def mlflow_transition_model(current_stage: str, new_stage: str):
    """Transition a model to a new stage in MLflow"""

    print("=== Transitioning model stage in MLflow ===")
    print(f"current stage: {current_stage}")
    print(f"new stage: {new_stage}")

    pass

    print("✅ Model stage transitioned in MLflow")
    print()
    return None

def mlflow_run(func):

    """Decorator to run a function within an MLflow run context"""
    print("=== Running function within MLflow run context ===")
    print(f"function: {func.__name__}")

    pass

    print("✅ Function run within MLflow context")
    print()
    return func
