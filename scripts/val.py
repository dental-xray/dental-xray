import kagglehub
from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Validation Script')
    parser.add_argument('--weights', type=str, required=True, help='model weights path')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='device (cpu or gpu id)')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    # parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    # parser.add_argument('--task', type=str, default='val', choices=['val', 'test'], help='validation or test')
    # parser.add_argument('--split', type=str, help='dataset split (overrides task)')
    # parser.add_argument('--save-txt', action='store_true', help='save results to txt files')
    # parser.add_argument('--save-json', action='store_true', help='save results to COCO JSON')
    # parser.add_argument('--save-hybrid', action='store_true', help='save hybrid results')
    # parser.add_argument('--project', type=str, default='runs/val', help='project name')
    # parser.add_argument('--name', type=str, default='exp', help='experiment name')
    # parser.add_argument('--verbose', action='store_true', help='verbose output')
    # parser.add_argument('--plots', action='store_true', help='save validation plots')

    return parser.parse_args()

def main():
    args = parse_args()

    print("=== YOLO Validation Script ===")
    print(f"Model: {args.weights}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")

    # Download latest version
    path = kagglehub.dataset_download("lokisilvres/dental-disease-panoramic-detection-dataset")
    print("Path to dataset files:", path)

    model = YOLO(args.weights)

    yaml_file = path + "/YOLO/YOLO/data.yaml"
    results = model.val(
            data=yaml_file, 
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
            plots=True
            )
    print(results.confusion_matrix.summary())
    print(results.confusion_matrix.to_df())


if __name__ == '__main__':
    main()

