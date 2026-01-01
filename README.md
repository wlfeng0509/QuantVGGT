
<div align="right">
  <details>
    <summary >🌐 Language</summary>
    <div>
      <div align="center">
        <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=en">English</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=zh-CN">简体中文</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=zh-TW">繁體中文</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=ja">日本語</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=ko">한국어</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=hi">हिन्दी</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=th">ไทย</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=fr">Français</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=de">Deutsch</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=es">Español</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=it">Italiano</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=ru">Русский</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=pt">Português</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=nl">Nederlands</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=pl">Polski</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=ar">العربية</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=fa">فارسی</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=tr">Türkçe</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=vi">Tiếng Việt</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=id">Bahasa Indonesia</a>
        | <a href="https://openaitx.github.io/view.html?user=wlfeng0509&project=QuantVGGT&lang=as">অসমীয়া</
      </div>
    </div>
  </details>
</div>

# Quantized Visual Geometry Grounded Transformer

[arXiv](https://arxiv.org/abs/2509.21302) | [BibTeX](#bibtex)

------

This project is the official implementation of our QuantVGGT: "Quantized Visual Geometry Grounded Transformer".

![teaser](imgs/teaser.png)

![overview](imgs/overview.png)

------

## Results

![result](imgs/result.png)

## Updates

- [October 10, 2025] Evaluation code for reproducing our camera pose estimation results on Co3D is now available.

## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub).

```
git clone git@github.com:wlfeng0509/QuantVGGT.git
cd QuantVGGT
pip install -r requirements.txt
```

Then download the pre trained weights provided by [VGGT](https://github.com/facebookresearch/vggt) and prepare Co3D dataset following [this](https://github.com/facebookresearch/vggt/tree/evaluation/evaluation).

Then download the pre trained W4A4 quantization parameters from [huggingface](https://huggingface.co/wlfeng/QuantVGGT/tree/main) and place the downloaded folder under *evaluation\outputs\w4a4* branch.

We can now use the provided script for inference **(remember to change the data path within the script)**.

```
cd evaluation
bash test.sh
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

