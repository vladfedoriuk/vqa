#!/bin/bash

python -m pip install -r requirements/base.txt .

python experiments/vilt/masked_language_modeling.py \
      --epochs 30 \
      --batch-size 16
