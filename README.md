# TTSR
Official PyTorch implementation of the paper [Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/abs/2006.04139) accepted in CVPR 2020.

## Contents
- [Introduction](#introduction)
  - [Contribution](#contribution)
  - [Approach overview](#approach-overview)
  - [Main results](#main-results)
- [Requirements and dependencies](#requirements-and-dependencies)
- [Model](#model)
- [Quick test](#quick-test)
- [Dataset prepare](#dataset-prepare)
- [Evaluation](#evaluation)
- [Train](#train)
- [Citation](#citation)
- [Contact](#contact)

## Introduction
We proposed an approach named TTSR for RefSR task. Compared to SISR, RefSR has an extra high-resolution reference image whose textures can be utilized to help super-resolve low-resolution input.

### Contribution
1. We are one of the first to introduce the transformer architecture into image generation tasks. More specifically, we propose a texture transformer with four closely-related modules for image SR which achieves significant improvements over SOTA approaches.
2. We propose a novel cross-scale feature integration module for image generation tasks which enables our approach to learn a more powerful feature representation by stacking multiple texture transformers.

### Approach overview
<img src="https://github.com/FuzhiYang/TTSR/blob/master/IMG/TT.png" width=40%><img src="https://github.com/FuzhiYang/TTSR/blob/master/IMG/CSFI.png" width=60%>

### Main results
<img src="https://github.com/FuzhiYang/TTSR/blob/master/IMG/results.png" width=80%>

## Requirements and dependencies
* python 3.7 (recommend to use [Anaconda](https://www.anaconda.com/))
* python packages: `pip install numpy opencv-python`
* pytorch >= 1.1.0
* torchvision >= 0.4.0

## Model
Pre-trained models can be downloaded from [onedrive](https://1drv.ms/u/s!Ajav6U_IU-1gmHZstHQxOTn9MLPh?e=e06Q7A), [baidu cloud](https://pan.baidu.com/s/1j9swBtz14WneuMYgTLkWtA)(0u6i), [google drive](https://drive.google.com/drive/folders/1CTm-r3hSbdYVCySuQ27GsrqXhhVOS-qh?usp=sharing).
* *TTSR-rec.pt*: trained with only reconstruction loss
* *TTSR.pt*: trained with all losses

## Quick test
1. Clone this github repo
```
git clone https://github.com/FuzhiYang/TTSR.git
cd TTSR
```
2. Download pre-trained models and modify "model_path" in test.sh
3. Run test
```
sh test.sh
```
4. The results are in "save_dir" (default: `./test/demo/output`)

## Dataset prepare
1. Download [CUFED train set](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I) and [CUFED test set](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)
2. Make dataset structure be:
- CUFED
    - train
        - input
        - ref
    - test
        - CUFED5

## Evaluation
1. Prepare CUFED dataset and modify "dataset_dir" in eval.sh
2. Download pre-trained models and modify "model_path" in eval.sh
3. Run evaluation
```
sh eval.sh
```
4. The results are in "save_dir" (default: `./eval/CUFED/TTSR`)

## Train
1. Prepare CUFED dataset and modify "dataset_dir" in train.sh
2. Run training
```
sh train.sh
```
3. The training results are in "save_dir" (default: `./train/CUFED/TTSR`)

## Citation
```
@InProceedings{yang2020learning,
author = {Yang, Fuzhi and Yang, Huan and Fu, Jianlong and Lu, Hongtao and Guo, Baining},
title = {Learning Texture Transformer Network for Image Super-Resolution},
booktitle = {CVPR},
year = {2020},
month = {June}
}
```

## Contact
If you meet any problems, please describe them in issues or contact:
* Fuzhi Yang: <yfzcopy0702@sjtu.edu.cn>

