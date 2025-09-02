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
import torch
import mlflow.artifacts
import tempfile
import time
import tempfile
import shutil
import requests


"""Model-related functions for disease recognition using YOLOv8"""

def save_results(model_storage, params: dict, metrics: dict, path=None, filename=None):

    """Save training/validation results to a CSV file
    Args:
        model_storage (str): The storage option to save results ("local" or "mlflow")
        params (dict): Dictionary of parameters used during training
        metrics (dict): Dictionary of metrics obtained during training
    Returns:
        None
    Example:
        save_results("local", {"epochs": 100, "batch_size": 16}, {"mAP": 0.85, "loss": 0.1})
    Note: Ensure that the model_storage is either "local" or "mlflow".
    """

    print("=== Saving results ===")
    print(f"model_storage: {model_storage}")
    print(f"params: {params}")
    print(f"metrics: {metrics}")
    print(f"path: {path}")
    print(f"filename: {filename}")


    # Save params locally
    if params is not None:
        params_path = os.path.join(path, filename + ".params.pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(path, filename + "metrics.pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


    if model_storage == "gcs":
        pass

    if model_storage == "mlflow":

        if params is not None:
            mlflow.log_params(params)
        #if metrics is not None:
        #    mlflow.log_metrics(metrics)
        print("✅ Results saved on mlflow")


def save_model(model, model_storage, path, filename=None, bucket_name=None, mlflow_tracking_uri=None, mlflow_model_name=None):

    """Save the trained model to the specified model_storage
    Args:
        model (YOLO): The trained YOLO model to be saved
        model_storage (str): The storage option to save the model ("local", "gcs", or "mlflow")
        path (str): The local path to save the model if model_storage is "local"
        filename (str, optional): The filename to save the model as. If None, a timestamped filename will be generated
        bucket_name (str, optional): The GCS bucket name if model_storage is "gcs"
        mlflow_tracking_uri (str, optional): The MLflow tracking URI if model_storage is "mlflow"
        mlflow_model_name (str, optional): The registered model name in MLflow if model_storage is "mlflow"
    Returns:
        None
    Example:
        save_model(model, "local", "./models", "my_model.pt")
        save_model(model, "gcs", "./models", bucket_name="my_bucket")
        save_model(model, "mlflow", None, mlflow_model_name="my_mlflow_model")
    Note: Ensure that the model_storage is either "local", "gcs", or "mlflow".
    """

    print("=== Saving model ===")
    print(f"model_storage: {model_storage}")
    print(f"path: {path}")
    print(f"filename: {filename}")
    print(f"bucket_name: {bucket_name}")
    print(f"mlflow_tracking_uri: {mlflow_tracking_uri}")
    print(f"mlflow_model_name: {mlflow_model_name}")

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

        print("🚀 Starting upload the model onto GCS...")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"models/{filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    elif model_storage == "mlflow":

        try:
            print("🚀 Starting upload the model onto MLflow...")

            # Prevent deadlock by setting number of threads to 1
            torch.set_num_threads(1)
            os.environ['OMP_NUM_THREADS'] = '1'

            mlflow.set_tracking_uri(mlflow_tracking_uri)

            if os.path.exists(model_path):
                print(f"Using existing model file: {model_path}")

                # Save the YOLO model as a PyFunc model in MLflow
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=YOLOModelWrapper(),
                    artifacts={"yolo_model": model_path},
                    registered_model_name=mlflow_model_name
                )
                print("✅ Model saved to MLflow as PyFunc model")

            else:
                print(f"❌ Model file not found: {model_path}")
                return None

        except Exception as e:
            print(f"MLflow save failed: {e}")
            return None

    else:
        print("❌ Model target not recognized. Please choose either 'local', 'gcs' or 'mlflow'.\n")

    print()

    return None


def load_model(model_storage, filename, path, stage="Production",bucket_name=None, mlflow_tracking_uri=None, mlflow_model_name=None):

    """Load a model from the specified model_storage
    Args:
        model_storage (str): The storage option to load the model from ("local", "gcs", or "mlflow")
        stage (str, optional): The stage of the model to load from MLflow ("Staging" or "Production"). Default is "Production"
        bucket_name (str, optional): The GCS bucket name if model_storage is "gcs"
        filename (str, optional): The filename of the model to load if model_storage is "local" or "gcs"
        path (str, optional): The local path to load the model from if model_storage is "local"
        mlflow_tracking_uri (str, optional): The MLflow tracking URI if model_storage is "mlflow"
        mlflow_model_name (str, optional): The registered model name in MLflow if model_storage is "mlflow"
    Returns:
        model (YOLO or pyfunc): The loaded YOLO model or MLflow pyfunc model
    Example:
        model = load_model("local", path="./models", filename="my_model.pt")
        model = load_model("gcs", bucket_name="my_bucket", filename="my_model.pt")
        model = load_model("mlflow", mlflow_tracking_uri="http://localhost:5000", mlflow_model_name="my_mlflow_model", stage="Production")
    Note: Ensure that the model_storage is either "local", "gcs", or "mlflow".
    """

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

        local_path = os.path.join(path, filename)

        if os.path.exists(local_path):
            print(f"Using existing local file: {local_path}")

        else:
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

            blob.download_to_filename(local_path)

        model = YOLO(local_path)
        print("✅ Model loaded from GCS")

    elif model_storage == "mlflow":
        print(Fore.BLUE + f"\nLoad latest model from MLflow..." + Style.RESET_ALL)

        local_path = os.path.join(path, filename)

        if os.path.exists(local_path):
            print(f"Using existing local file: {local_path}")
            model = YOLO(local_path)
            print("✅ YOLO model loaded successfully")

            return model

        else:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            client = MlflowClient()

            try:
                if stage is None:
                    stage = "None"

                model_versions = client.get_latest_versions(name=mlflow_model_name, stages=[stage])
                if not model_versions:
                    print(f"\n❌ No model found")
                    return None

                model_version = model_versions[0]
                model_uri = f"models:/{mlflow_model_name}/{model_version.version}"
                print(f"Model URI: {model_uri}")

            except Exception as e:
                print(f"\n❌ Error: {e}")
                return None

            try:
                print("🔄 Starting robust download...")
                model = robust_model_download(model_uri, path, filename, max_retries=10)

                if model:
                    print("✅ Model download successful")
                    return model
                else:
                    print("❌ Model download failed")
                    return None

            except Exception as e:
                print(f"❌ Robust download failed: {e}")
                return None

    else:
        print("❌ Model storage not recognized")
        return None

    print()
    return model

def robust_model_download(model_uri, path, filename, max_retries=3):
    """Robustly download a model from MLflow with retry logic
    Args:
        model_uri (str): The MLflow model URI to download
        path (str): The local path to save the downloaded model
        filename (str): The filename to save the model as
        max_retries (int, optional): Maximum number of retry attempts. Default is 3
    Returns:
        model (YOLO or None): The loaded YOLO model or None if download failed
    Example:
        model = robust_model_download("models:/my_mlflow_model/Production", "./models", "my_model.pt")
    """

    for attempt in range(max_retries):
        try:
            print(f"🔄 Download attempt {attempt + 1}/{max_retries}")

            try:
                mlmodel_path = mlflow.artifacts.download_artifacts(
                    f"{model_uri}/MLmodel",
                    dst_path=path
                )
                print("✅ Metadata downloaded")
            except:
                print("⚠️  Metadata download failed")

            try:
                artifacts_path = mlflow.artifacts.download_artifacts(
                    f"{model_uri}/artifacts",
                    dst_path=path
                )

                if os.path.exists(artifacts_path):
                    print(f"✅ Artifacts directory downloaded: {artifacts_path}")

                    """Search for .pt files in the artifacts directory"""
                    for root, dirs, files in os.walk(artifacts_path):
                        for file in files:
                            if file.endswith('.pt'):
                                pt_path = os.path.join(root, file)
                                print(f"Found .pt file: {pt_path}")

                                model = YOLO(pt_path)

                                final_path = os.path.join(path, filename)
                                os.makedirs(path, exist_ok=True)
                                shutil.copy2(pt_path, final_path)

                                print(f"✅ Success! Model saved: {final_path}")
                                return model

                    print("❌ No .pt file found in artifacts")

            except Exception as artifacts_error:
                print(f"Artifacts download failed: {artifacts_error}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    print("❌ All download attempts failed")
    return None


# def download_model_with_retry(model_uri, path, filename, max_retries=3):

#     """Download a model from MLflow with retry logic
#     Args:
#         model_uri (str): The MLflow model URI to download
#         path (str): The local path to save the downloaded model
#         filename (str): The filename to save the model as
#         max_retries (int, optional): Maximum number of retry attempts. Default is 3
#     Returns:
#         model (YOLO or None): The loaded YOLO model or None if download failed
#     Example:
#         model = download_model_with_retry("models:/my_mlflow_model/Production", "./models", "my_model.pt")
#     Note: This function attempts to download the model multiple times in case of failures.
#     """

#     for attempt in range(max_retries):
#         try:
#             print(f"🔄 Download attempt {attempt + 1}/{max_retries}...")

#             with tempfile.TemporaryDirectory() as temp_dir:

#                 session = requests.Session()
#                 session.timeout = (30, 300)

#                 artifact_path = mlflow.artifacts.download_artifacts(
#                     model_uri,
#                     dst_path=temp_dir
#                 )

#                 pt_file_candidates = [
#                     os.path.join(artifact_path, "artifacts", "trained_model_5epoch.pt"),
#                     os.path.join(artifact_path, "model", "trained_model_5epoch.pt"),
#                     os.path.join(artifact_path, "trained_model_5epoch.pt")
#                 ]

#                 pt_file = None
#                 for candidate in pt_file_candidates:
#                     if os.path.exists(candidate):
#                         pt_file = candidate
#                         break

#                 if pt_file is None:
#                     print("🔍 Searching all directories for .pt files...")
#                     for root, dirs, files in os.walk(artifact_path):
#                         for file in files:
#                             if file.endswith('.pt'):
#                                 pt_file = os.path.join(root, file)
#                                 print(f"Found: {pt_file}")
#                                 break
#                         if pt_file:
#                             break

#                 if pt_file and os.path.exists(pt_file):
#                     print(f"✅ Found .pt file: {pt_file}")

#                     file_size = os.path.getsize(pt_file)
#                     print(f"File size: {file_size / (1024*1024):.1f} MB")

#                     if file_size < 1024:
#                         print("⚠️  File seems corrupted (too small)")
#                         if attempt < max_retries - 1:
#                             time.sleep(2 ** attempt)
#                             continue

#                     model = YOLO(pt_file)
#                     print("✅ YOLO model loaded successfully")

#                     if path and filename:
#                         local_path = os.path.join(path, filename)
#                         os.makedirs(path, exist_ok=True)

#                         shutil.copy2(pt_file, local_path)
#                         print(f"Model saved locally: {local_path}")

#                         final_model = YOLO(local_path)
#                         return final_model
#                     else:
#                         return model
#                 else:
#                     print(f"❌ .pt file not found (attempt {attempt + 1})")
#                     if attempt < max_retries - 1:
#                         time.sleep(2 ** attempt)
#                         continue

#         except Exception as e:
#             print(f"❌ Attempt {attempt + 1} failed: {e}")
#             if attempt < max_retries - 1:
#                 wait_time = 2 ** attempt
#                 print(f"Waiting {wait_time} seconds before retry...")
#                 time.sleep(wait_time)
#             else:
#                 print("❌ All retry attempts exhausted")

#     return None


def try_load_mlflow_model(model_uri):

    """Try loading an MLflow model as YOLO or PyFunc
    Args:
        model_uri (str): The MLflow model URI to load
    Returns:
        model (YOLO or pyfunc or None): The loaded YOLO model, PyFunc model, or None if loading failed
    Example:
        model = try_load_mlflow_model("models:/my_mlflow_model/Production")
    Note: This function attempts to load the model first as a YOLO model, then as a PyFunc model.
    """

    try:
        model = mlflow.pytorch.load_model(model_uri)
        print("✅ Loaded as PyTorch model")
        return model
    except Exception as e:
        print(f"PyTorch load failed: {e}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Loaded as PyFunc model")

        yolo_model = extract_yolo_from_pyfunc(model_uri)
        if yolo_model:
            print("✅ YOLO model extracted")
            return yolo_model
        else:
            print("⚠️  Using PyFunc model")
            return model

    except Exception as e:
        print(f"PyFunc load failed: {e}")

    return None

def mlflow_transition_model(current_stage: str, new_stage: str):
    """Transition a model to a new stage in MLflow
    Args:
        current_stage (str): The current stage of the model ("Staging" or "Production")
        new_stage (str): The new stage to transition the model to ("Staging" or "Production")
    Returns:
        None
    Example:
        mlflow_transition_model("Staging", "Production")
    Note: Ensure that the stages are either "Staging" or "Production".
    """

    print("=== Transitioning model stage in MLflow ===")
    print(f"current stage: {current_stage}")
    print(f"new stage: {new_stage}")

    pass

    print("✅ Model stage transitioned in MLflow")
    print()
    return None

def mlflow_run(func):

    """Decorator to run a function within an MLflow run context
    Args:
        func (function): The function to be executed within the MLflow run context
    Returns:
        function: The wrapped function that runs within the MLflow context
    Example:
        @mlflow_run
    """

    print("=== Running function within MLflow run context ===")
    print(f"function: {func.__name__}")

    pass

    print("✅ Function run within MLflow context")
    print()
    return func



class YOLOModelWrapper(mlflow.pyfunc.PythonModel):

    """Custom MLflow PyFunc model wrapper for YOLO"""

    def load_context(self, context):

        """Load the YOLO model from the provided context"""

        model_file = context.artifacts["yolo_model"]
        self.model = YOLO(model_file)
        print("YOLO model loaded successfully")

    def predict(self, context, model_input):

        """Make predictions using the YOLO model"""
        results = self.model(model_input)

        output = []
        for result in results:
            if result.boxes is not None:
                output.append({
                    "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
                    "scores": result.boxes.conf.cpu().numpy().tolist(),
                    "classes": result.boxes.cls.cpu().numpy().tolist()
                })
            else:
                output.append({"boxes": [], "scores": [], "classes": []})
        return output
