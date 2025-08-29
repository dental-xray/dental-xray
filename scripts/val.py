from ultralytics import YOLO
import argparse
import sys
from disease_recognition.params import *
from disease_recognition.data import load_data


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

    print("=== YOLO Validation Script ===", file=sys.stderr)
    print(f"Weights: {args.weights}", file=sys.stderr)
    print(f"Batch size: {args.batch}", file=sys.stderr)
    print(f"Image size: {args.imgsz}", file=sys.stderr)
    print(f"Device: {args.device}", file=sys.stderr)

    # Download latest version
    path = load_data()

    model = YOLO(args.weights)

    results = model.val(
            data=DATA_FILE,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
            plots=True
            )

    print(results.confusion_matrix.to_csv())
    print(f"mAP@0.5: {results.box.map50}", file=sys.stderr)
    print(f"mAP@0.5:0.95: {results.box.map}", file=sys.stderr)
    print(f"Precision: {results.box.mp}", file=sys.stderr)
    print(f"Recall: {results.box.mr}", file=sys.stderr)


if __name__ == '__main__':
    main()
