#!/bin/bash

 for shot in 2 8 16 32; do for seed in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do python CMA_fewshot.py --seed $seed --dataset_name 'politifact' --train_csv "./datasets/fakenewsnet/politifact_multi.csv" --img_path "./datasets/fakenewsnet/poli_img_all/" --shot $shot --save_path "./saved_adapter"; done; done
