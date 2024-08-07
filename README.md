# RFFNet: Towards Robust and Flexible Fusion for Low-Light Image Denoising (ACM MM 2024)

Qiang Wang, Yuning Cui, Yawen Li, Yaping Ruan, Ben Zhu, [Wenqi Ren](https://scholar.google.com.hk/citations?user=VwfgfR8AAAAJ&hl=zh-CN&oi=ao)

[![](https://img.shields.io/badge/ACM%20MM-paper-blue.svg)]()

## Abstract

>Low-light environments will introduce high-intensity noise into images. Containing fine details with reduced noise, near-infrared/flash images can serve as guidance to facilitate noise removal. 
However, existing fusion-based methods fail to effectively suppress artifacts caused by inconsistency between guidance/noisy image pairs and do not fully excavate the useful information contained in guidance images. In this paper, we propose a robust and flexible fusion network (RFFNet) for low-light image denoising. Specifically, we present a multi-scale inconsistency calibration module to address inconsistency before fusion by first mapping the guidance features to multi-scale spaces and calibrating them with the aid of pre-denoising features in a coarse-to-fine manner. Furthermore, we develop a dual-domain adaptive fusion module to adaptively extract useful high-/low-frequency signals from the guidance features and then highlight the informative frequencies.
Extensive experimental results demonstrate that our method achieves state-of-the-art performance on NIR-guided RGB image denoising and flash-guided no-flash image denoising.

## Network Architecture
![fig](./fig/intro_archi.png)

## Dataset Preparation
### DVD Dataset
Please download the dataset from [link](https://drive.google.com/drive/folders/10FV0q_GAP4gjQUbQ78waezfyGO07AxlP?usp=share_link) first. 

Then, unzip the file into `Dataset` directory.
And the directory structure is organized as:

```
Dataset
├── DVD_train_raw
│     ├── RGB
│     ├── NIR
├── DVD_test
│     ├── RGB
│     ├── NIR
├── DVD_real
│     ├── RGB
│     ├── NIR
```

Finally, create the training patches for faster data loading by

`python generate_train_patches.py`

And the directory structure now is organized as:

```
Dataset
├── DVD_train_raw
│     ├── RGB
│     ├── NIR
├── DVD_train
│     ├── RGB
│     ├── NIR
├── DVD_test
│     ├── RGB
│     ├── NIR
├── DVD_real
│     ├── RGB
│     ├── NIR
```
### FAID Dataset
...

## Train And Evaluation
```bash
git clone (*代码发布时补充*)
cd DVN

export PYTHONPATH="${PYTHONPATH}:./"
pip install requirements.txt

# For the AutoEncoder (i.e. the network AE that provides supervision signals for DSEM) training, 
# set MODEL.MODE in training.yml to `Recons`, and run:
python train.py

# After the AutoEncoder (i.e. AE) training, you can visualize the deep structure supervision signals of RGB/NIR by:
python visualization/view_edgedetect.py

# For the DVN training, 
# set MODEL.MODE in training.yml to `Fusion`, and run:
python train.py

# You can evaluate the performance of DVN by:
python test.py

# Or you can evaluate the performance of DVN by:
python test_real.py

# You can visualize the calculated DIP by run:
python visualization/view_dip.py
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
## Contact
Should you have any questions, please contact Qiang Wang.
