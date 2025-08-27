#!/usr/bin/bash

DEVICE="cuda"
BATCH=12
IMGSZ=420

MODEL="best.pt"
echo "evaluate $MODEL"
python val.py --weights ../models/${MODEL} --device ${DEVICE} --batch $BATCH --imgsz $IMGSZ> ${MODEL}.val

MODEL="trained_model.pt"
echo "evaluate $MODEL"
python val.py --weights ../models/${MODEL} --device ${DEVICE} --batch $BATCH --imgsz $IMGSZ > ${MODEL}.val

MODEL="trained_model_100epoch.pt"
echo "evaluate $MODEL"
python val.py --weights ../models/${MODEL} --device ${DEVICE} --batch $BATCH --imgsz $IMGSZ > ${MODEL}.val
