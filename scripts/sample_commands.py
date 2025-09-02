import argparse
from disease_recognition.params import *
from disease_recognition.data import *
from disease_recognition.model import *
from disease_recognition.registry import *


path = load_data(DATASET)

(model, _results) = train_model(
    data=DATA_FILE,
    weights=WEIGHTS_TRAIN,
    device=DEVICE,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE
)

params = {
    "data": DATA_FILE,
    "weights": WEIGHTS_TRAIN,
    "device": DEVICE,
    "epochs": EPOCHS,
    "imgsz": IMG_SIZE,
    "batch": BATCH_SIZE
}

results = model.predict("/Users/syatsuzuka/.cache/kagglehub/datasets/lokisilvres/dental-disease-panoramic-detection-dataset/versions/6/YOLO/YOLO/test/images/ff050195-SABAQI_HASAN_2020-07-01203554_jpg.rf.c136b8d6f429857f7d335252645c646b.jpg")
print(results[0].boxes)
save_model(model, model_storage="mlflow", path=LOCAL_REGISTRY_PATH, filename="test19.pt", bucket_name=BUCKET_NAME, mlflow_tracking_uri=MLFLOW_TRACKING_URI, mlflow_model_name=MLFLOW_MODEL_NAME)
save_results(model_storage="mlflow", params=params, metrics=_results.metrics, path=LOCAL_REGISTRY_PATH, filename="test19.pt")


#======= Check registered models in MLflow =======#
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
all_experiments = client.search_experiments()
print(all_experiments)
