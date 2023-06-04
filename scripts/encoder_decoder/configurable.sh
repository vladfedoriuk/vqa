#!/bin/bash

python -m pip install -r requirements/base.txt .

python experiments/encoder_decoder/configurable.py \
      --epochs 15 \
      --batch-size 16 \
      --dataset "vqa_v2" \
      --encoder-decoder-backbone "google/vit-base-patch16-224-in21k-bert-base-uncased"
