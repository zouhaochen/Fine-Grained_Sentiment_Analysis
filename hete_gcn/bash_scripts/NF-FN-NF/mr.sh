#!/bin/bash
python train.py --verbose  --data_dir datasets --dataset mr --path NF-FN-NF --wt_reg 10.000000 --learning_rate 0.002000 --dropout 0.250000 --emb_reg 0.000000 --FN_norm None --NF_norm sym