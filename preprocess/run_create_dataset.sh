#!/bin/bash

python create_dataset.py \
  --labels-dir "/Users/lucas/Documents/Projetos/Multiple-Choise-Recognition/preprocess/output" \
  --images-dir "/Users/lucas/Documents/Projetos/Datasets/tamaulipas/imgs_Tam_dataset" \
  --dataset-dir "./dataset_tamaulipas_base" \
  --valid-size 0.2 \
  --class-names A B C D