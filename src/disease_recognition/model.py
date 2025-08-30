import pandas as pd
from ultralytics import YOLO
from disease_recognition.params import *
from datetime import datetime


def init_model (input_shape: tuple):

    """Initialize the YOLO model with the given input shape"""

    print("=== Initializing model ===")
    print(f"input_shape: {input_shape}")

    pass

    print("✅ Model initialized")
    return None

def train_model (data: str, weights="yolov8n.pt", device="cpu", epochs=100, batch=2, imgsz=420, patience=50):

    """Train the YOLO model with the given parameters"""
    print("=== Training model ===")
    print(f"data: {data}")
    print(f"weights: {weights}")
    print(f"device: {device}")
    print(f"epochs: {epochs}")
    print(f"batch: {batch}")
    print(f"imgsz: {imgsz}")
    print(f"patience: {patience}")

    model = YOLO(weights)
    model.info()

    results = model.train(
        data=data,
        device=device,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience
        #cache=True,
        #lr0=0.001,
        #optimizer='AdamW',
        #mosaic=0.5,
        #mixup=0.0,
        #degrees=5.0,
        #translate=0.05
    )

    save_dir = results.save_dir
    print(f"✅ Results saved in: {save_dir}")

    return model, results


def val_model (data: str, weights, device="cpu", batch=12, imgsz=420):

    """Validate the YOLO model with the given parameters"""
    print("=== Validating model ===")
    print(f"data: {data}")
    print(f"weights: {weights}")
    print(f"device: {device}")
    print(f"batch: {batch}")
    print(f"imgsz: {imgsz}")

    model = YOLO(weights)

    results = model.val(
        data=data,
        device=device,
        batch=batch,
        imgsz=imgsz,
        plots=True
    )

    print(results.confusion_matrix.to_csv())
    print(f"mAP@0.5: {results.box.map50}", file=sys.stderr)
    print(f"mAP@0.5:0.95: {results.box.map}", file=sys.stderr)
    print(f"Precision: {results.box.mp}", file=sys.stderr)
    print(f"Recall: {results.box.mr}", file=sys.stderr)

    print("✅ Validation completed")
    print()

    return results


def pred_model (model: YOLO, image_file: str):
    """Predict using the YOLO model on a given image file"""

    print("=== Predicting ===")
    print(f"image_file: {image_file}")

    pass

    print("✅ Prediction completed")
    print()

    return None
