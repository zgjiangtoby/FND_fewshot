#!/bin/bash

for shot in 2 4 6 8 16 32 64 100 128; do for seed in 1 2 3 4 5; do python adapter_fewshot.py --seed $seed --dataset_name 'politifact' --train_csv "./datasets/fakenewsnet/politifact_multi.csv" --img_path "./datasets/fakenewsnet/poli_img_all/" --shot $shot --save_path "./saved_adapter"; done; done

