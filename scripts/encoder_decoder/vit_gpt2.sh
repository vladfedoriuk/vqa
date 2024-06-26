#!/bin/bash

python -m pip install -r requirements/base.txt .

python experiments/encoder_decoder/vit_gpt2.py \
      --epochs 100 \
      --batch-size 16
