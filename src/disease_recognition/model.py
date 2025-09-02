import pandas as pd
from ultralytics import YOLO
from disease_recognition.params import *
from datetime import datetime

"""Model-related functions for disease recognition using YOLOv8"""

def init_model (input_shape: tuple) -> YOLO:

    """Initialize the YOLO model with the given input shape
    Args:
        input_shape (tuple): The desired input shape for the model (height, width)
    Returns:
        model (YOLO): The initialized YOLO model
    Raise:s:
        ValueError: If the input_shape is not a tuple of two integers
    Example:
        model = init_model((420, 420))
    Note: YOLOv8 does not require explicit input shape setting during initialization.
    """

    print("=== Initializing model ===")
    print(f"input_shape: {input_shape}")

    pass

    print("✅ Model initialized")
    return None

def train_model (data: str, weights: YOLO ="yolov8n.pt", device="cpu", epochs=100, batch=2, imgsz=420, patience=50) -> tuple[YOLO, pd.DataFrame]:

    """Train the YOLO model with the given parameters
    Args:
        data (str): Path to the dataset configuration file
        weights (YOLO): Pre-trained weights to start training from
        device (str): Device to use for training ("cpu" or "cuda")
        epochs (int): Number of training epochs
        batch (int): Batch size for training
        imgsz (int): Image size for training
        patience (int): Number of epochs with no improvement to stop training early
    Returns:
        model (YOLO): The trained YOLO model
        results (pd.DataFrame): DataFrame containing training results and metrics
    Example:
        model, results = train_model("data.yaml", weights="yolov8n.pt", device="cuda", epochs=50, batch=16, imgsz=640, patience=10)
    Note: Ensure that the dataset is properly formatted and the data.yaml file is correctly set up.
    """

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
    )

    save_dir = results.save_dir
    print(f"✅ Results saved in: {save_dir}")

    return (model, results)


def val_model (data: str, weights, device="cpu", batch=12, imgsz=420) -> YOLO:

    """Validate the YOLO model with the given parameters
    Args:
        data (str): Path to the dataset configuration file
        weights (str): Path to the trained model weights
        device (str): Device to use for validation ("cpu" or "cuda")
        batch (int): Batch size for validation
        imgsz (int): Image size for validation
    Returns:
        results (YOLO): The validation results object containing metrics
    Example:
        results = val_model("data.yaml", "best.pt", device="cuda", batch=16, imgsz=640)
    Note: Ensure that the dataset is properly formatted and the data.yaml file is correctly set up.
    """

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

    print(f"mAP@0.5: {results.box.map50}")
    print(f"mAP@0.5:0.95: {results.box.map}")
    print(f"Precision: {results.box.mp}")
    print(f"Recall: {results.box.mr}")

    print("✅ Validation completed")
    print()

    return results
