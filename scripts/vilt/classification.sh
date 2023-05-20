#!/bin/bash

python -m pip install -r requirements/base.txt .

python experiments/vilt/classification.py \
      --vilt-backbone "dandelin/vilt-b32-finetuned-vqa" \
      --dataset "daquar" \
      --epochs 100 \
      --batch-size 16
