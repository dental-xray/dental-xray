import os
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)


#======= General settings =======#
PROJECT_ROOT=os.environ.get("PROJECT_ROOT")
DATASET=os.environ.get("DATASET","lokisilvres/dental-disease-panoramic-detection-dataset")
DATA_FILE=os.environ.get("DATA_FILE")
MODEL_STORAGE=os.environ.get("MODEL_STORAGE","local")
LOCAL_REGISTRY_PATH=os.environ.get("LOCAL_REGISTRY_PATH",os.path.join(PROJECT_ROOT,"models"))
SCRIPT_DIR=os.environ.get("SCRIPT_DIR",os.path.join(PROJECT_ROOT,"scripts"))
WORK_DIR=os.environ.get("WORK_DIR",os.path.join(PROJECT_ROOT,".tmp"))

#======= Variables for training =======#
WEIGHTS_TRAIN = os.environ.get("WEIGHTS_TRAIN")
WEIGHTS_PREDICT = os.environ.get("WEIGHTS_PREDICT")
EPOCHS = int(os.environ.get("EPOCHS", 100))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
DEVICE = os.environ.get("DEVICE", "0")
LEARNING_RATE = os.environ.get("LEARNING_RATE", 0.01)
MOMENTUM = os.environ.get("MOMENTUM", 0.937)
WEIGHT_DECAY = os.environ.get("WEIGHT_DECAY", 0.0005)
AUGMENTATION = os.environ.get("AUGMENTATION", "True").lower()

#======= Variables for GCP =======#
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
# GAR_IMAGE = os.environ.get("GAR_IMAGE")
# GAR_MEMORY = os.environ.get("GAR_MEMORY")

#======= Variables for MLflow =======#
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
MLFLOW_RUN_NAME = os.environ.get("MLFLOW_RUN_NAME")
