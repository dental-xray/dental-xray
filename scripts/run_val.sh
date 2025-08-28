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


MODEL="best.pt"
echo "evaluate $MODEL..."
WEIGHTS="~/.cache/kagglehub/datasets/lokisilvres/dental-disease-panoramic-detection-dataset/versions/6/${MODEL}"
CSV_FILE="../models/${MODEL}.val.csv"
LOG_FILE="../models/${MODEL}.val.log"
python val.py --weights ${WEIGHTS} --device ${DEVICE} --batch ${BATCH} --imgsz $IMGSZ 1> ${CSV_FILE} 2> ${LOG_FILE}

MODEL="trained_model.pt"
echo "evaluate $MODEL..."
WEIGHTS="../models/${MODEL}"
CSV_FILE="../models/${MODEL}.val.csv"
LOG_FILE="../models/${MODEL}.val.log"
python val.py --weights ${WEIGHTS} --device ${DEVICE} --batch ${BATCH} --imgsz ${IMGSZ} 1> ${CSV_FILE} 2> ${LOG_FILE}

MODEL="trained_model_100epoch.pt"
echo "evaluate $MODEL..."
WEIGHTS="../models/${MODEL}"
CSV_FILE="../models/${MODEL}.val.csv"
LOG_FILE="../models/${MODEL}.val.log"
python val.py --weights ${WEIGHTS} --device ${DEVICE} --batch ${BATCH} --imgsz ${IMGSZ} 1> ${CSV_FILE} 2> ${LOG_FILE}
