CUDA_VISIBLE_DEVICES=0 python make_calibation.py \
    --model_path ../VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir /wmq/3d_datasets/ \
    --co3d_anno_dir /wmq/datasets/co3d_v2_annotations/ \
    --seed 0 \
    --each_nsamples 10 \
    --cache_path ./outputs/source_calib_data.pt \
    --save_path ./outputs/total_20_calib_data.pt \
    --class_mode all \
    --kmeans_n 5 \
    --kmeans_m 4 \




