#!/usr/bin/bash

################################################################################
#  File:  run_val.sh
#  Desc:  Run validation script
################################################################################

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <DEVICE> [WEIGHTS] [EPOCH] [DEVICE] [BATCH_SIZE] [IMAGE_SIZE]"
  echo
  echo "  DEVICE: 'cpu' or 'cuda'"
  echo "  WEIGHTS: Model file to use for training (default: yolo11n-seg.pt)"
  echo "  EPOCH: Number of epochs to train (default: 100)"
  echo "  BATCH_SIZE: Batch size for validation (default: 2)"
  echo "  IMAGE_SIZE: Image size for validation (default: 420)"
  echo
  exit 1
fi

DEVICE=$1

if [ "$2" != "" ]; then
    WEIGHTS=$2
else
    WEIGHTS="yolo11n-seg.pt"
fi

if [ "$3" != "" ]; then
    EPOCH=$3
else
    EPOCH=100
fi

if [ "$4" != "" ]; then
    BATCH=$4
else
    BATCH=2
fi

if [ "$5" != "" ]; then
    IMGSZ=$5
else
    IMGSZ=420
fi

OUTPUT_FILE="trained_model_${EPOCH}epoch.pt"
LOG_FILE="${OUTPUT_FILE}.train.log"
VAL_LOG_FILE="${OUTPUT_FILE}.val.log"

echo "Training model..."
python train.py --weights ${WEIGHTS} --device ${DEVICE} --epochs ${EPOCH} --batch-size ${BATCH} --imgsz ${IMGSZ} > ${LOG_FILE} 2>&1

echo "Evaluating model..."
python val.py --weights ${OUTPUT_FILE} --device ${DEVICE} --batch ${BATCH} --imgsz ${IMGSZ} > ${VAL_LOG_FILE} 2>&1

echo "Moving model to models directory..."
mv ${OUTPUT_FILE} ../models/
mv ${LOG_FILE} ../models/
mv ${VAL_LOG_FILE} ../models/
