#!/bin/bash

for dataset in daquar vqa_v2_sample
  do
    for fusion in concatenation-fusion attention-fusion
      do
        for image_encoder_backbone in facebook/dino-vitb16 facebook/deit-base-distilled-patch16-224 microsoft/beit-base-patch16-224-pt22k
          do
            for text_encoder_backbone in bert-base-uncased distilbert-base-uncased roberta-base
              do
                python experiments/fusion/classification.py \
                    --image-encoder-backbone "$image_encoder_backbone" \
                    --text-encoder-backbone "$text_encoder_backbone" \
                    --fusion "$fusion" \
                    --dataset "$dataset" \
                    --epochs 100
              done
          done
      done
  done
