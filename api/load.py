import os
from ultralytics import YOLO
from google.cloud import storage
#from mlflow.tracking import MlflowClient --- IGNORE ---
#import mlflow --- IGNORE ---


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
        model = YOLO(os.path.join(path, filename))

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

    # elif model_storage == "mlflow":

    #     print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

    #     mlflow.set_tracking_uri(mlflow_tracking_uri)
    #     client = MlflowClient()

    #     try:
    #         if stage is None:
    #             stage = "None"

    #         model_versions = client.get_latest_versions(name=mlflow_model_name, stages=[stage])
    #         if not model_versions:
    #             print(f"\n❌ No model found")
    #             return None

    #         model_version = model_versions[0]
    #         model_uri = f"models:/{mlflow_model_name}/{model_version.version}"
    #         print(f"Model URI: {model_uri}")

    #     except Exception as e:
    #         print(f"\n❌ Error: {e}")
    #         return None

    #     # Ignore PyFunc and directly get the .pt file
    #     try:
    #         print("🔄 Downloading artifacts...")

    #         with tempfile.TemporaryDirectory() as temp_dir:
    #             artifact_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=temp_dir)

    #             # Search for the .pt file (it should be in artifacts/trained_model_5epoch.pt)
    #             pt_file = os.path.join(artifact_path, "artifacts", "trained_model_5epoch.pt")

    #             if os.path.exists(pt_file):
    #                 print(f"✅ Found .pt file: {pt_file}")
    #                 model = YOLO(pt_file)
    #                 print("✅ YOLO model loaded successfully")

    #                 # Save the model locally with a timestamped filename
    #                 local_path = f"./mlflow_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    #                 model.save(local_path)
    #                 print(f"Model saved locally: {local_path}")

    #             else:
    #                 print("❌ .pt file not found at expected location")
    #                 return None

        # except Exception as e:
        #     print(f"❌ Model load failed: {e}")
        #     return None

    else:
        print("❌ Model storage not recognized")
        return None

    print()
    return model
