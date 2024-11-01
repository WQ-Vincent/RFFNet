# RFFNet: Towards Robust and Flexible Fusion for Low-Light Image Denoising (ACM MM 2024)

Qiang Wang, Yuning Cui, Yawen Li, Yaping Ruan, Ben Zhu, Wenqi Ren

[![paper](https://img.shields.io/badge/ACM%20MM-paper-blue.svg)](https://dl.acm.org/doi/10.1145/3664647.3680675)

## Abstract

>Low-light environments will introduce high-intensity noise into images. Containing fine details with reduced noise, near-infrared/flash images can serve as guidance to facilitate noise removal. 
However, existing fusion-based methods fail to effectively suppress artifacts caused by inconsistency between guidance/noisy image pairs and do not fully excavate the useful information contained in guidance images. In this paper, we propose a robust and flexible fusion network (RFFNet) for low-light image denoising. Specifically, we present a multi-scale inconsistency calibration module to address inconsistency before fusion by first mapping the guidance features to multi-scale spaces and calibrating them with the aid of pre-denoising features in a coarse-to-fine manner. Furthermore, we develop a dual-domain adaptive fusion module to adaptively extract useful high-/low-frequency signals from the guidance features and then highlight the informative frequencies.
Extensive experimental results demonstrate that our method achieves state-of-the-art performance on NIR-guided RGB image denoising and flash-guided no-flash image denoising.

## Network Architecture
![fig](./fig/arch.png)

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1, and CUDA 11.1.
For installing, follow these instructions:
~~~
conda create -n rffnet python=3.8 && conda activate rffnet

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
~~~

## Dataset Preparation
Please download the public dataset from [DVD](https://drive.google.com/drive/folders/10FV0q_GAP4gjQUbQ78waezfyGO07AxlP?usp=share_link). 

Then, unzip the file into `dataset/DVD` directory.
The directory structure is organized as:

```
DVD
├── train_raw
│     ├── RGB
│     ├── NIR
├── test
│     ├── RGB
│     ├── NIR
```

Finally, create the training patches for faster data loading by

`python generate_train_patches.py`

The directory structure is now organized as:

```
DVD
├── train_raw
│     ├── RGB
│     ├── NIR
├── train
│     ├── RGB
│     ├── NIR
├── test
│     ├── RGB
│     ├── NIR
```


## Citation
If you find this project useful for your research, please consider citing:
~~~
@inproceedings{wang2024rffnet,
  title={RFFNet: Towards Robust and Flexible Fusion for Low-Light Image Denoising},
  author={Wang, Qiang and Cui, Yuning and Li, Yawen and Ruan, Yaping and Zhu, Ben and Ren, Wenqi},
  booktitle={ACM Multimedia},
  year={2024}
}
~~~
## Acknowledgements
This repository is greatly inspired by [DVN](https://github.com/megvii-research/DVN).
