from ultralytics import YOLO
import argparse
from disease_recognition.params import *
from disease_recognition.data import *
from disease_recognition.model import *


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    parser.add_argument('--weights', type=str, default=WEIGHTS_TRAIN, help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model config file')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='image size')
    parser.add_argument('--device', type=str, default=DEVICE, help='device (cpu or gpu id)')
    # parser.add_argument('--workers', type=int, default=8, help='number of workers')
    # parser.add_argument('--project', type=str, default='runs/train', help='project name')
    # parser.add_argument('--name', type=str, default='exp', help='experiment name')
    # parser.add_argument('--resume', action='store_true', help='resume training')
    # parser.add_argument('--save-period', type=int, default=10, help='save checkpoint every x epochs')

    return parser.parse_args()


def main():

    args = parse_args()

    print(WEIGHTS_TRAIN)
    print(DATASET)

    print("=== YOLO Training Script ===")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")

    path = load_data(DATASET)

    model = train_model(
        weights=args.weights,
        data=DATA_FILE,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size
    )

    save_model(model)

if __name__ == '__main__':
    main()
