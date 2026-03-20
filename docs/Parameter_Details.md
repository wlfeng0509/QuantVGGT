**Parameter Details**

**make_calibation.py**

```
Parameter list:
- --model_path : Path to VGGT model
- --co3d_dir : CO3D dataset directory
- --co3d_anno_dir : CO3D annotation directory
- --seed : Random seed
- --cache_path : Path to source dataset cache. If it does not exist, it will select class_mode categories from CO3D, with each_nsamples per category
- --save_path : Path to save the calibration dataset, which is a subset of the source dataset
- --class_mode : Category mode for the source dataset
- --each_nsamples : Number of samples per category
- --kmeans_n : Number of cluster centers
- --kmeans_m : Number of samples per cluster
```

**run_co3d.py** 

```
Parameter list:
- --model_path : Path to original VGGT model
- --co3d_dir : CO3D dataset directory
- --co3d_anno_dir : CO3D annotation directory
- --dtype : Quantization data type
- --seed : Random seed
- --lac : Enable LAC (Layer-wise Accumulation Calibration)
- --lwc : Enable LWC (Layer-wise Weight Calibration)
- --cache_path : Path to filtered calibration dataset. If it exists and is valid, each_nsamples will be ignored
- --class_mode : Category mode for the default calibration dataset and evaluation
- --each_nsamples : Number of samples per category for the default calibration dataset
- --exp_name : Experiment name, very important, corresponds to the quantized model name. If resume_qs is True, it will load directly
- --fast_eval : Fast evaluation mode, only evaluate the first 20 samples per category
- --resume_qs : Load existing quantized weights corresponding to exp_name
```

**run_7andN.py** 

```
Parameter list:
- --model_path : Path to original VGGT model
- --co3d_dir : CO3D dataset directory
- --co3d_anno_dir : CO3D annotation directory
- --class_mode : Category mode for the default calibration dataset
- --each_nsamples : Number of samples per category for the default calibration dataset
- --dtype : Quantization data type
- --lwc : Enable LWC (Layer-wise Weight Calibration)
- --lac : Enable LAC (Layer-wise Accumulation Calibration)
- --cache_path : Path to filtered calibration dataset
- --output_dir : Directory to save evaluation results
- --kf : Frame sampling interval
- --dataset : Type of dataset to evaluate, 7s for 7-Scenes, NR for NRGBD
- --dataset_path : Path to 7-Scenes or NRGBD dataset
- --exp_name : Experiment name
- --resume_qs : Load existing quantized weights corresponding to exp_name
```