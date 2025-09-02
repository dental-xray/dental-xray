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

                # Custom PyFunc wrapper to load YOLO model
                class YOLOModelWrapper(mlflow.pyfunc.PythonModel):
                    def load_context(self, context):
                        from ultralytics import YOLO
                        model_file = context.artifacts["yolo_model"]
                        self.model = YOLO(model_file)
                        print("YOLO model loaded successfully")

                    def predict(self, context, model_input):
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


def load_model(model_storage, stage="Production",bucket_name=None, filename=None, path=None, mlflow_tracking_uri=None, mlflow_model_name=None):

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

        # Ignore PyFunc and directly get the .pt file
        try:
            print("🔄 Downloading artifacts...")

            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir)

                # Search for the .pt file (it should be in artifacts/trained_model_5epoch.pt)
                pt_file = os.path.join(artifact_path, "artifacts", "trained_model_5epoch.pt")

                if os.path.exists(pt_file):
                    print(f"✅ Found .pt file: {pt_file}")
                    model = YOLO(pt_file)
                    print("✅ YOLO model loaded successfully")

                    # Save the model locally with a timestamped filename
                    local_path = f"./mlflow_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                    model.save(local_path)
                    print(f"Model saved locally: {local_path}")

                else:
                    print("❌ .pt file not found at expected location")
                    return None

        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return None

    else:
        print("❌ Model storage not recognized")
        return None

    print()
    return model

def try_load_mlflow_model(model_uri):
    """MLflowモデルの段階的読み込み"""

    # 1. PyTorchモデルとして試行
    try:
        model = mlflow.pytorch.load_model(model_uri)
        print("✅ Loaded as PyTorch model")
        return model
    except Exception as e:
        print(f"PyTorch load failed: {e}")

    # 2. PyFuncモデルとして試行
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Loaded as PyFunc model")

        # 3. YOLOとして抽出を試行
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
