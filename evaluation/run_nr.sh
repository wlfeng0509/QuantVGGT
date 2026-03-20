CUDA_VISIBLE_DEVICES=0 python run_7andN.py\
    --model_path ../VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir /wmq/3d_datasets/ \
    --co3d_anno_dir /wmq/datasets/co3d_v2_annotations/ \
    --class_mode all\
    --each_nsamples 10 \
    --dtype quarot_w4a4\
    --lwc \
    --lac \
    --cache_path ./outputs/cache_data.pt \
    --output_dir "eval_results" \
    --kf 100 \
    --dataset nr \
    --dataset_path /wmq/dataset/neural_rgbd_data \
    --exp_name a44 \
    --resume_qs


