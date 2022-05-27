#!/bin/bash

function train {
    run_file='train.py'
    path_config='config_'$name'.yaml'
    path_data=$folder_data'/'$name'_'$idx'_'$sub_idx'.h5'
    path_viewpoint=$folder_data'/'$name'_viewpoint_'$idx'_'$sub_idx'.h5'
    folder_log='experiments/logs/'$name'_'$idx'_'$sub_idx
    folder_out='experiments/outs/'$name'_'$idx'_'$sub_idx
    python $run_file \
        --input_dir 'input_dir' \
        --output_dir 'output_dir' \
        --batch_size 4 \
        --optimiser Adam \
        --lr_rate 0.0002 \
        --seed 0 \
        --num_slots 8 \
        --pixel_sigma 0.1 \
        --temperature 0.0 \
        --latent_dim 16 \
        --view_dim 5 \
        --min_sample_views 1 \
        --max_sample_views 3 \
        --num_vq_show 4 \
        --query_nll 1.0 \
        --exp_nll 1.0 \
        --exp_attention 1.0 \
        --kl_latent 1.0 \
        --kl_spatial 1.0 \
        --path_config $path_config \
        --path_data $path_data \
        --path_viewpoint $path_viewpoint \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
}

function test {
    run_file='eval.py'
    path_config='config_'$name'.yaml'
    path_data=$folder_data'/'$name'_'$idx'_'$sub_idx'.h5'
    path_viewpoint=$folder_data'/'$name'_viewpoint_'$idx'_'$sub_idx'.h5'
    folder_log='experiments/logs/'$name'_'$idx'_'$sub_idx
    folder_out='experiments/outs/'$name'_'$idx'_'$sub_idx
    python $run_file \
        --input_dir 'input_dir' \
        --output_dir 'output_dir' \
        --batch_size 1 \
        --optimiser Adam \
        --lr_rate 0.0002 \
        --seed 0 \
        --num_slots 8 \
        --pixel_sigma 0.1 \
        --temperature 0.0 \
        --latent_dim 16 \
        --view_dim 5 \
        --min_sample_views 1 \
        --max_sample_views 3 \
        --num_vq_show 4 \
        --query_nll 1.0 \
        --exp_nll 1.0 \
        --exp_attention 1.0 \
        --kl_latent 1.0 \
        --kl_spatial 1.0 \
        --path_config $path_config \
        --path_data $path_data \
        --path_viewpoint $path_viewpoint \
        --folder_log $folder_log \
        --folder_out $folder_out
}

folder_data='../multiple-unspecified-viewpoints/data'

for name in 'clevr_multi' 'shop_multi'; do
    for idx in 1 2; do
        for sub_idx in 1 2; do
            train
            test
        done
    done
done
