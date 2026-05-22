#!/bin/bash

python3 ./create_dataset.py \
  --labels-dir "/Users/lucas/Documents/Projetos/Multiple-Choise-Recognition/preprocess/tamaulipas/outputs" \
  --dataset-dir "/Users/lucas/Documents/Projetos/Datasets/dataset_tamaulipas_base" \
  --valid-size 0.2 \
  --class-names A B C D 
