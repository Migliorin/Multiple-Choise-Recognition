#!/bin/bash

python infer_yolo.py \
  --data /Users/lucas/Documents/Projetos/Datasets/dataset_kaggle_base/data.yaml \
  --model /Users/lucas/Documents/Projetos/Multiple-Choise-Recognition/runs/detect/runs_yolo/tamaulipas_yolo11n-2/weights/best.pt \
  --imgsz 640 \
  --conf 0.4 \
  --n 4