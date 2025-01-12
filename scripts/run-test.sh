#!/bin/bash -v
python main_ad.py --anormly_ratio 0.9 -ep 3 --batch_size 128 --mode test --data_name PSM --win_size 105 --stride 1 --patch_size [3,5,7] --patch_mx 0.1 --d_model 60 --e_layer 3 -lr 0.0001 --data_path /data/zaggyai/DCdetector_dataset

