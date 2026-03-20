# Quantized Visual Geometry Grounded Transformer

<p align="center">
  <a href="https://arxiv.org/abs/2509.21302">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square" alt="Paper"/>
  </a>
  <a href="https://github.com/facebookresearch/vggt">
    <img src="https://img.shields.io/badge/GitHub-Code-blue?style=flat-square&logo=github" alt="Code"/>
  </a>
  <a href="https://huggingface.co/wlfeng/QuantVGGT">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?style=flat-square" alt="Model"/>
  </a>
</p>

------

<div align="center">
Weilun Feng1,2∗, Haotong Qin3∗, Mingqiang Wu1,2∗, Chuanguang Yang1†, Yuqi Li1, Xiangqi Li1,2, Zhulin An1†, Libo Huang1, Yulun Zhang4, Michele Magno3, Yongjun Xu1 

</div>

<sup>*</sup>Equal Contribution  <sup>†</sup>Corresponding Author，

1.Institute of Computing Technology, Chinese Academy of Sciences，

2.University of Chinese Academy of Sciences，

3.ETH Z¨ urich,

4.Shanghai Jiao Tong University

</div>

## 📰 News

- **[2026.01.26]** 😉 The paper has been accepted by ICLR 2026.

- **[2025.10.10]** 🎉 Paper and code released! Check out our [paper]([Quantized Visual Geometry Grounded Transformer](https://arxiv.org/pdf/2509.21302)).

## 🚀 Updates

- **[2026.03.20]** 😉Added Gradio demo script for web-based online execution, supplemented the missing eval directory, and fixed known bugs.
- **[2026.02.09]** 😉The calibration dataset and quantized weights have been updated in the Hugging Face repository. Please [check](https://huggingface.co/wlfeng/QuantVGGT).
- **[2026.02.08]** 🎉Code for calibration training, evaluation on the 7-Scene and NRGBD datasets, and calibration set selection is now available.
- **[2025.10.10]** 🎉Evaluation code for reproducing our camera pose estimation results on Co3D is now available.



![teaser](imgs/teaser.png)

![overview](imgs/overview.png)

------



## 🌟Results

![result](imgs/result.png)



## 🛠️ Installation

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub).

```
git clone git@github.com:wlfeng0509/QuantVGGT.git
cd QuantVGGT
pip install -r requirements.txt
pip install -r requirements_demo.txt
```

Then download the pre trained weights provided by [VGGT](https://github.com/facebookresearch/vggt) and prepare Co3D dataset following [this](https://github.com/facebookresearch/vggt/tree/evaluation/evaluation).

Then download the pre trained W4A4 quantization parameters from [huggingface](https://huggingface.co/wlfeng/QuantVGGT/tree/main) and place the downloaded folder under ***evaluation\outputs\w4a4*** branch.

Then download the calibration set from [huggingface](https://huggingface.co/wlfeng/QuantVGGT/tree/main) and place the downloaded folder under ***evaluation\outputs*** branch.

## 🤖 Online Demo [NEW]

If the quantized weights come from our repository [huggingface](https://huggingface.co/wlfeng/QuantVGGT/tree/main), please run directly:

```
bash /demo_gradio.sh
```

Otherwise, modify `exp_name` and `dtype` to the names you specified, then run.

```
python demo_gradio.py\
    --device "cuda:0" \
    --model_path ./VGGT-1B/model_tracker_fixed_e20.pt \
    --lwc \
    --lac \
    --exp_name YOUR_NAME \
    --dtype YOUR_BITS\
```

## 📊 Quick start

We can now use the provided script for inference **(remember to change the data path within the script)**.

**[1-1 simple] Filter and Save Co3d Calibration Set**

```
cd evaluation
bash make_calibation.sh 
```

**[1-2 detailed] Filter and Save Co3d Calibration Set**

If `cache_path` is missing, it will automatically build `class_mode * each_nsamples` as the source dataset.

```
python make_calibation.py \
    --model_path ../VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir /data1/3d_datasets/ \
    --co3d_anno_dir /data2/fwl/datasets/co3d_v2_annotations/ \
    --seed 0 \
    --cache_path outputs/source_calib_data.pt \  # Path to the source dataset
    --save_path outputs/total_20_calib_data.pt \ # Path to the filtered calibration dataset
    --class_mode all \                           # Type of classes for building the source dataset
    --each_nsamples 10 \                         # Number of samples per class for building the source dataset
    --kmeans_n 5 \                               # Number of cluster centers for the calibration dataset
    --kmeans_m 4 \                               # Number of samples per cluster for the calibration dataset
```

**[2-1 simple] Quantize calibrate and evaluate on Co3d.**

If the quantized weights and calibration set weights come from our repository [huggingface](https://huggingface.co/wlfeng/QuantVGGT/tree/main), please run:

```
bash run_co3d.sh
```

**[2-2 detailed] Quantize calibrate and evaluate on Co3d.**

```
python run_co3d.py \
    --model_path ../VGGT-1B/model_tracker_fixed_e20.pt \ # Path to original VGGT
    --co3d_dir 3d_datasets/ \
    --co3d_anno_dir co3d_v2_annotations/ \
    --dtype quarot_w4a4 \
    --seed 0 \
    --lac \
    --lwc \
    --cache_path ./outputs/cache_data.pt \ # Filtered calibration dataset
    --class_mode all \                     # Number of classes in the default calibration dataset
    --each_nsamples 10 \                   # Number of samples per class in the default calibration dataset
    --exp_name a44 \
    --fast_eval \                          # Only evaluate the first 20 samples per class
    --resume_qs                            # Load existing weights for exp_name
```

**[3] Quantize calibrate on Co3d, evaluate on 7 scenes or NRGBD dataset.**

Please download the 7 Scenes and NRGBD datasets, then modify the path of `dataset_path`.

```
bash run_7s.sh
```

```
bash run_nr.sh
```

Or

```
python run_7andN.py \
    --model_path ../VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir 3d_datasets/ \
    --co3d_anno_dir co3d_v2_annotations/ \
    --class_mode all \
    --each_nsamples 10 \
    --dtype quarot_w4a4 \
    --lwc \
    --lac \
    --cache_path ./outputs/cache_data.pt \
    --output_dir "./eval_results" \
    --kf 100 \                     # Frame sampling interval
    --dataset 7s \                 # Evaluate 7-Scenes or NRGBD dataset
    --dataset_path 7-Scenes/data \ # Path to 7-Scenes or NRGBD dataset
    --exp_name a44 \
    --resume_qs
```

Also, you can use the quantized model for predicting other 3D attributes following the guidance [here](https://github.com/facebookresearch/vggt/tree/evaluation#detailed-usage).

If you are confused about the above parameters, please refer to the detailed instructions [Parameter Details](/docs/Parameter Details)

## 🥸 Comments

* Our codebase is heavily builds on [VGGT](https://github.com/facebookresearch/vggt) and [QuaRot](https://github.com/spcl/QuaRot). Thanks for open-sourcing!

## 😍 BibTeX

If you find *QuantVGGT* is useful and helpful to your work, please kindly cite this paper:

```
@article{feng2025quantized,
  title={Quantized Visual Geometry Grounded Transformer},
  author={Feng, Weilun and Qin, Haotong and Wu, Mingqiang and Yang, Chuanguang and Li, Yuqi and Li, Xiangqi and An, Zhulin and Huang, Libo and Zhang, Yulun and Magno, Michele and others},
  journal={arXiv preprint arXiv:2509.21302},
  year={2025}
}
```

## 📧Contact

For questions or suggestions, please open an issue or contact:

- Weilun Feng: [fengweilun24s@ict.ac.cn](https://github.com/wlfeng0509/Fast-SAM3D/blob/main/fengweilun24s@ict.ac.cn)
- Mingqiang Wu [wumingqiang25e@ict.ac.cn](mailto:wumingqiang25e@ict.ac.cn)
- Chuanguang Yang: [yangchuanguang@ict.ac.cn](mailto:yangchuanguang@ict.ac.cn)
- Zhulin An: [anzhulin@ict.ac.cn](mailto:anzhulin@ict.ac.cn)
