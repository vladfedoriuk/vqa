#!/bin/bash

python -m pip install -r requirements/base.txt .

python experiments/vilt/masked_language_modeling.py \
      --epochs 50 \
      --batch-size 16
