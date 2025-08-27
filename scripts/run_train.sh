#!/usr/bin/bash

MODEL="yolo11n-seg.pt"
EPOCH=100
DEVICE="cuda"
BATCH=2
IMGSZ=416

echo "Training model..."
python train.py --weights ${MODEL} --device ${DEVICE} --epochs ${EPOCH} --batch-size ${BATCH} --imgsz ${IMGSZ} > result_train.txt

echo "Evaluating model..."
python val.py --weights ../models/${MODEL} --device ${DEVICE} --batch $BATCH --imgsz $IMGSZ > trained_model.pt.val
