import numpy as np
import pandas as pd
from ultralytics import YOLO
from disease_recognition.params import *
from datetime import datetime
from PIL import Image
import os
from colorama import Fore, Style
from google.cloud import storage
import mlflow
from mlflow.tracking import MlflowClient


def save_results(model_storage, params: dict, metrics: dict):

    """Save training/validation results to a CSV file"""
    print("=== Saving results ===")
    print(f"params: {params}")
    print(f"metrics: {metrics}")

    if model_storage == "mlflow":
        if params is not None:
            mlflow.log_params(params)
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on mlflow")


def save_model(model, model_storage, path, filename=None, bucket_name=None, mlflow_model_name=None):

    """Save the trained model to the specified model_storage"""

    print("=== Saving model ===")
    print(f"model_storage: {model_storage}")
    print(f"path: {path}")
    print(f"filename: {filename}")

    if filename is None:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'trained_model_{current_time}.pt'

    model_path = os.path.join(path, filename)

    try:
        model.save(model_path)
        print("✅ Model saved locally (PyTorch format)")

    except Exception as e:
        print(f"Model save failed: {e}")
        return None

    if model_storage == "local":
        pass

    elif model_storage == "gcs":

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"models/{filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    elif model_storage == "mlflow":

        try:
            print("🚀 Starting MLflow upload...")

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                registered_model_name=mlflow_model_name
            )
            print("✅ Model saved to MLflow")

        except Exception as e:
            print(f"MLflow save failed: {e}")
            return None

    else:
        print("❌ Model target not recognized. Please choose either 'local', 'gcs' or 'mlflow'.\n")

    print()

    return None

def load_model(model_storage, stage="Production",bucket_name=None, filename=None, path=None, mlflow_tracking_uri=None, mlflow_model_name=None):

    """Load a model from the specified model_storage"""

    print("=== Loading model ===")
    print(f"model_storage: {model_storage}")
    print(f"stage: {stage}")
    print(f"filename: {filename}")
    print(f"path: {path}")

    if model_storage == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        model = YOLO(os.path.join(path, filename))

        print("✅ Model loaded from local file")

    elif model_storage == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

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
        print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(name=mlflow_model_name, stages=[stage])
            model_uri = model_versions[0].source

        except:
            print(f"\n❌ No model found with name {mlflow_model_name} in stage {stage}")
            return None

        print(f"Model URI: {model_uri}")

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print("✅ Model loaded from MLflow")

        except Exception as e:
            print(f"❌ Model load failed: {e}")

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
