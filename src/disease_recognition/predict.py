from dataclasses import dataclass
from typing import Tuple
from ultralytics import YOLO
import os


@dataclass
class BoxDetection:
    classification: int
    confidence: float
    boxes: list


@dataclass
class MaskDetection:
    xy: list[Tuple[float, float]]  # List of mask xy coordinates


@dataclass
class PredictionResult:
    box: list[BoxDetection]
    mask: list[MaskDetection]


def predict(model, image_path: str) -> PredictionResult:
    """
    Run inference on a single image and return detections.
    Each detection contains class, confidence, and bounding box.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    results = model(image_path)
    result = results[0]
    return PredictionResult(
        box=[
            BoxDetection(
                classification=int(box.cls),
                confidence=float(box.conf),
                boxes=box.xyxy.tolist(),
            )
            for box in result.boxes
        ],
        mask=[
            MaskDetection(
                xy=[unpack_coordinates(maskxy) for maskxy in mask.xy[0].tolist()]
            )
            for mask in result.masks
        ],
    )


def unpack_coordinates(coords) -> Tuple[float, float]:
    """
    Unpack coordinates from a tensor-like structure to a tuple of floats.
    """
    return float(coords[0]), float(coords[1])
