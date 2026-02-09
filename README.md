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

<sup>*</sup>Equal Contribution  <sup>†</sup>Corresponding Author

1.Institute of Computing Technology, Chinese Academy of Sciences, 2.University of Chinese Academy of Sciences,3.ETH Z¨ urich, 4.Shanghai Jiao Tong University

</div>

## 📰 News

- **[2026.01.26]** 😉 The paper has been accepted by ICLR 2026.

- **[2025.10.10]** 🎉 Paper and code released! Check out our [paper]([Quantized Visual Geometry Grounded Transformer](https://arxiv.org/pdf/2509.21302)).

## 🚀 Updates

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

## 📊 Quick start

We can now use the provided script for inference **(remember to change the data path within the script)**.

```
cd evaluation
bash make_calibation.sh # Filter and Save Calibration Set
bash run_co3d.sh # Calibration Training and Evaluation on Co3D
```

**Generate filtered Co3d calibration data**

```
python Quant_VGGT/vggt/evaluation/make_calibation.py \
    --model_path VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir co3d_datasets/ \
    --co3d_anno_dir co3d_v2_annotations/ \
    --seed 0 \
    --cache_path all_calib_data.pt \ # Data to be filtered for calibration set
    --save_path calib_data.pt \ # Save path for calibration set
    --class_mode all \  # Category selection mode for calibration data
    --kmeans_n 6 \ # Number of cluster centers
    --kmeans_m 7 \  # Number of samples per category
```

**Quantize calibrate and evaluate on Co3d.**

```
python Quant_VGGT/vggt/evaluation/run_co3d.py \
    --model_path Quant_VGGT/VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir co3d_datasets/ \
    --co3d_anno_dir co3d_v2_annotations/ \
    --dtype quarot_w4a4\	# Quantization Bit Width
    --seed 0 \
    --lac \ 
    --lwc \
    --cache_path calib_data.pt \ # calibration data path
    --class_mode all \	# Category selection mode for calibration data
    --exp_name a44_uqant \	
    --resume_qs \ # Load quantized model from exp_name
```

**Quantize calibrate on Co3d, evaluate on 7 scenes or NRGBD dataset.**

```
python Quant_VGGT/vggt/evaluation/run_7andN.py\
  --model_path Quant_VGGT/VGGT-1B/model_tracker_fixed_e20.pt \
  --co3d_dir co3d_datasets/ \
  --co3d_anno_dir co3d_v2_annotations/ \
  --dtype quarot_w4a4\
  --lwc \
  --lac \
  --exp_name quant_w4a4\
  --cache_path calib_data.pt \
  --class_mode all \
  --output_dir "Quant_VGGT/vggt/eval_results" \
  --kf 100 \ # Sample every keyframe.
  --dataset nr \ # evaluation 7s or nr dataset
```





Also, you can use the quantized model for predicting other 3D attributes following the guidance [here](https://github.com/facebookresearch/vggt/tree/evaluation#detailed-usage).



## Comments

* Our codebase is heavily builds on [VGGT](https://github.com/facebookresearch/vggt) and [QuaRot](https://github.com/spcl/QuaRot). Thanks for open-sourcing!

## BibTeX

If you find *QuantVGGT* is useful and helpful to your work, please kindly cite this paper:

```
@article{feng2025quantized,
  title={Quantized Visual Geometry Grounded Transformer},
  author={Feng, Weilun and Qin, Haotong and Wu, Mingqiang and Yang, Chuanguang and Li, Yuqi and Li, Xiangqi and An, Zhulin and Huang, Libo and Zhang, Yulun and Magno, Michele and others},
  journal={arXiv preprint arXiv:2509.21302},
  year={2025}
}
```

