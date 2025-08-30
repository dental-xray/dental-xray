#!/bin/zsh

################################################################################
#  File:  run_val.sh
#  Desc:  Run validation script
################################################################################

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <OUTPUT_FILE> [DEVICE] [WEIGHTS_TRAIN] [EPOCH] [DEVICE] [BATCH_SIZE] [IMAGE_SIZE]"
  echo
  echo "Arguments:"
  echo "  OUTPUT_FILE: Output file to save the trained model (e.g., trained_model.pt)"
  echo "  DEVICE: 'cpu' or 'cuda'"
  echo "  WEIGHTS_TRAIN: Model file to use for training (default: yolo11n-seg.pt)"
  echo "  EPOCH: Number of epochs to train (default: 100)"
  echo "  BATCH_SIZE: Batch size for validation (default: 2)"
  echo "  IMAGE_SIZE: Image size for validation (default: 420)"
  echo
  exit 1
fi

source .env

OUTPUT_FILE=$1

if [ "$2" != "" ]; then
  DEVICE=$2
fi

if [ "$3" != "" ]; then
    WEIGHTS_TRAIN=$3
fi

if [ "$4" != "" ]; then
    EPOCHS=$4
fi

if [ "$5" != "" ]; then
    BATCH_SIZE=$5
fi

if [ "$6" != "" ]; then
    IMG_SIZE=$6
fi

TRAIN_LOG_FILE="${OUTPUT_FILE}.train.log"
VAL_LOG_FILE="${OUTPUT_FILE}.val.log"
VAL_CSV_FILE="${OUTPUT_FILE}.val.csv"

echo "Output file: ${OUTPUT_FILE}"
echo "Device: ${DEVICE}"
echo "Weights for training: ${WEIGHTS_TRAIN}"
echo "Epoch: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${IMG_SIZE}"
echo "Train log file: ${TRAIN_LOG_FILE}"
echo "Validation CSV file: ${VAL_CSV_FILE}"
echo "Validation log file: ${VAL_LOG_FILE}"
echo

rm -rf ${WORK_DIR}
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

echo "Training model..."
COMMAND="python ${SCRIPT_DIR}/train.py --output ${OUTPUT_FILE} --weights ${WEIGHTS_TRAIN} --device ${DEVICE} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --imgsz ${IMG_SIZE} > ${TRAIN_LOG_FILE}"
echo "COMMAND: $COMMAND"
eval $COMMAND

echo
echo "Evaluating model..."
COMMAND="python ${SCRIPT_DIR}/val.py --output ${VAL_CSV_FILE} --weights ${OUTPUT_FILE} --device ${DEVICE} --batch ${BATCH_SIZE} --imgsz ${IMG_SIZE} > ${VAL_LOG_FILE}"
echo "COMMAND: $COMMAND"
eval $COMMAND

echo
echo "Moving model to models directory..."
COMMAND="mv ${OUTPUT_FILE} ${TRAIN_LOG_FILE} ${VAL_LOG_FILE} ${VAL_CSV_FILE} ${LOCAL_REGISTRY_PATH}"
echo "COMMAND: $COMMAND"
eval $COMMAND
