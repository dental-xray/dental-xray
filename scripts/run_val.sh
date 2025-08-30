#!/bin/zsh

################################################################################
#  File:  run_val.sh
#  Desc:  Run validation script
################################################################################

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <TARGET_MODEL> [DEVICE] [BATCH_SIZE] [IMAGE_SIZE]"
  echo
  echo "Arguments:"
  echo "  TARGET_MODEL: Model file to use for validation (e.g., all or trained_model.pt)"
  echo "  DEVICE: 'cpu' or 'cuda'"
  echo "  BATCH_SIZE: Batch size for validation (default: 12)"
  echo "  IMAGE_SIZE: Image size for validation (default: 420)"
  echo
  exit 1
fi

source .env

TARGET_MODEL=$1

if [ "$2" != "" ]; then
  DEVICE=$2
fi

if [ "$3" != "" ]; then
  BATCH_SIZE=$3
fi

if [ "$4" != "" ]; then
  IMG_SIZE=$4
fi

echo "Target model: ${TARGET_MODEL}"
echo "Device: ${DEVICE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Image size: ${IMG_SIZE}"
echo


rm -rf ${WORK_DIR}
cd ${WORK_DIR}

if [ "${TARGET_MODEL}" = "all" ]; then

  MODEL="best.pt"
  echo "evaluate $MODEL..."
  WEIGHTS="${HOME}/.cache/kagglehub/datasets/lokisilvres/dental-disease-panoramic-detection-dataset/versions/6/${MODEL}"
  CSV_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.csv"
  LOG_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.log"
  COMMAND="python ${SCRIPT_DIR}/val.py --weights ${WEIGHTS} --device ${DEVICE} --batch ${BATCH_SIZE} --imgsz ${IMG_SIZE} 1> ${CSV_FILE} 2> ${LOG_FILE}"
  echo "COMMAND: $COMMAND"
  eval $COMMAND

  echo
  MODEL="trained_model_5epoch.pt"
  echo "evaluate $MODEL..."
  WEIGHTS="${LOCAL_REGISTRY_PATH}/${MODEL}"
  CSV_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.csv"
  LOG_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.log"
  COMMAND="python ${SCRIPT_DIR}/val.py --weights ${WEIGHTS} --device ${DEVICE} --batch ${BATCH_SIZE} --imgsz ${IMG_SIZE} 1> ${CSV_FILE} 2> ${LOG_FILE}"
  echo "COMMAND: $COMMAND"
  eval $COMMAND

  echo
  MODEL="trained_model_100epoch.pt"
  echo "evaluate $MODEL..."
  WEIGHTS="${LOCAL_REGISTRY_PATH}/${MODEL}"
  CSV_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.csv"
  LOG_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.log"
  COMMAND="python ${SCRIPT_DIR}/val.py --weights ${WEIGHTS} --device ${DEVICE} --batch ${BATCH_SIZE} --imgsz ${IMG_SIZE} 1> ${CSV_FILE} 2> ${LOG_FILE}"
  echo "COMMAND: $COMMAND"
  eval $COMMAND

else

  echo "Evaluate model: ${TARGET_MODEL}"

  MODEL=${TARGET_MODEL}
  echo "evaluate $MODEL..."
  WEIGHTS="${LOCAL_REGISTRY_PATH}/${MODEL}"
  CSV_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.csv"
  LOG_FILE="${LOCAL_REGISTRY_PATH}/${MODEL}.val.log"
  COMMAND="python ${SCRIPT_DIR}/val.py --weights ${WEIGHTS} --device ${DEVICE} --batch ${BATCH_SIZE} --imgsz ${IMG_SIZE} 1> ${CSV_FILE} 2> ${LOG_FILE}"
  echo "COMMAND: $COMMAND"
  eval $COMMAND

fi
