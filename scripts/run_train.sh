#!/usr/bin/bash

################################################################################
#  File:  run_val.sh
#  Desc:  Run validation script
################################################################################

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <DEVICE> [BATCH_SIZE] [IMAGE_SIZE]"
  echo
  echo "  DEVICE: 'cpu' or 'cuda'"
  echo "  BATCH_SIZE: Batch size for validation (default: 12)"
  echo "  IMAGE_SIZE: Image size for validation (default: 420)"
  echo
  exit 1
fi

DEVICE=$1

if [ "$2" != "" ]; then
    BATCH=$2
else
    BATCH=12
fi

if [ "$3" != "" ]; then
    IMGSZ=$3
else
    IMGSZ=420
fi

MODEL="yolo11n-seg.pt"
EPOCH=100
DEVICE="cuda"
BATCH=2
IMGSZ=416

echo "Training model..."
python train.py --weights ${MODEL} --device ${DEVICE} --epochs ${EPOCH} --batch-size ${BATCH} --imgsz ${IMGSZ} > result_train.txt

echo "Evaluating model..."
python val.py --weights ../models/${MODEL} --device ${DEVICE} --batch $BATCH --imgsz $IMGSZ > trained_model.pt.val
