import kagglehub
from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    # parser.add_argument('--cfg', type=str, default='', help='model config file')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='device (cpu or gpu id)')
    # parser.add_argument('--workers', type=int, default=8, help='number of workers')
    # parser.add_argument('--project', type=str, default='runs/train', help='project name')
    # parser.add_argument('--name', type=str, default='exp', help='experiment name')
    # parser.add_argument('--resume', action='store_true', help='resume training')
    # parser.add_argument('--save-period', type=int, default=10, help='save checkpoint every x epochs')

    return parser.parse_args()


def main():

    args = parse_args()

    print("=== YOLO Training Script ===")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")


    # Download latest version
    path = kagglehub.dataset_download("lokisilvres/dental-disease-panoramic-detection-dataset")

    print("Path to dataset files:", path)


    # Load a COCO-pretrained YOLOv8n model
    model = YOLO(args.weights)

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    yaml_file = path + "/YOLO/YOLO/data.yaml"
    results = model.train(
        data=yaml_file,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        #cache=True,                   # RAMキャッシュ
        #patience=50,
        #lr0=0.001,
        #optimizer='AdamW',
        #mosaic=0.5,
        #mixup=0.0,
        #degrees=5.0,
        #translate=0.05
    )

    save_dir = results.save_dir
    print(f"Results saved in: {save_dir}")

    # best_model = results.save_dir / 'weights' / 'best.pt'
    model.save(f'trained_model_{args.epochs}epochs.pt')

if __name__ == '__main__':
    main()
