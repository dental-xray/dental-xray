from ultralytics import YOLO
import argparse
import sys
from disease_recognition.params import *
from disease_recognition.data import load_data
from disease_recognition.model import val_model
import os

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Validation Script')
    parser.add_argument('--output', type=str, default="trained_model.pt.val.csv", help='file to save confusion matrix')
    parser.add_argument('--weights', type=str, required=True, help='model weights path')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='device (cpu or gpu id)')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')

    return parser.parse_args()

def main():

    """Main function to validate the YOLO model"""

    args = parse_args()

    print("=== YOLO Validation Script ===")
    print(f"Output: {args.output}")
    print(f"Weights: {args.weights}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print()

    load_data(DATASET)

    weights = os.path.join(LOCAL_REGISTRY_PATH, args.weights)

    results = val_model(
        data=DATA_FILE,
        weights=weights,
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz
    )

    results.confusion_matrix.to_csv(args.output)

if __name__ == '__main__':
    main()
