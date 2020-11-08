#!/bin/bash

# makde sure the below path are correctly configured before you run this script
data_path=<YOUR-PATH>/mulmon_datasets/clevr                                            # <---------- configure your own path to the data here
repo_path=.
log_path=${repo_path}/logs
data_type=clevr_aug   # one of ["clevr_aug", "clevr_mv"]

python train.py --arch mv_MulMON --datatype ${data_type} --work_mode training \
--input_dir ${data_path} --output_dir ${log_path} \
--batch_size 1 --epochs 2000 --step_per_epoch 300 --optimiser Adam --lr_rate 0.0001 --seed 0 \
--num_slots 7 --pixel_sigma 0.1 --temperature 0.0 --latent_dim 16 --view_dim 5 --min_sample_views 1 --max_sample_views 6 --num_vq_show 5 \
--query_nll 1.0 --exp_nll 1.0 --exp_attention 1.0 --kl_latent 1.0 --kl_spatial 1.0 \
--nodes 1 --gpus 1 --gpu_start 1 --master_port '29500' --use_bg
