import pandas as pd
from ultralytics import YOLO
from disease_recognition.params import *


def init_model (input_shape: tuple):
    pass

def train_model (data: str, weights="yolo8n.pt", device="cpu", epochs=100, batch=2, imgsz=420, patience=50):

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
    print(f"Results saved in: {save_dir}")

    return model, results


def save_model (model: YOLO):

    output_path = f'trained_model_{args.epochs}epoch.pt'
    model.save(output_path)

    print(f"Trained model saved to: {output_path}")

    return output_path


def val_model (data: str, weights, device="cpu", batch=12, imgsz=420):

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

    return results


def pred_model (model: YOLO, image_file: str) -> list:
    pass
