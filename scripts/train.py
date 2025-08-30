from ultralytics import YOLO
import argparse
from disease_recognition.params import *
from disease_recognition.data import *
from disease_recognition.model import *
from disease_recognition.registry import *

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    parser.add_argument('--output', type=str, default="trained_model.pt", help='output model file')
    parser.add_argument('--weights', type=str, default=WEIGHTS_TRAIN, help='initial weights path')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='image size')
    parser.add_argument('--device', type=str, default=DEVICE, help='device (cpu or gpu id)')


    return parser.parse_args()


def main():

    """Main function to train the YOLO model"""

    args = parse_args()

    print(WEIGHTS_TRAIN)
    print(DATASET)

    print("=== YOLO Training Script ===")
    print(f"Output: {args.output}")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print()

    path = load_data(DATASET)

    (model, _results) = train_model(
        data=DATA_FILE,
        weights=args.weights,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size
    )

    save_model(model, storage="local", path=LOCAL_REGISTRY_PATH, filename=args.output)

if __name__ == '__main__':
    main()
