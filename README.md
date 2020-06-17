# TTSR
Official pytorch implementation of the paper [Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/abs/2006.04139) accepted in CVPR 2020.

## Requirements and dependencies
* Python 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/))
* Python packages: `pip install numpy opencv-python`
* Pytorch >= 1.1.0
* Torchvision >= 0.4.0

## Model
Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1CTm-r3hSbdYVCySuQ27GsrqXhhVOS-qh?usp=sharing).
* *TTSR-rec.pt*: trained with only reconstruction loss
* *TTSR.pt*: trained with all losses

## Evaluation
1. Clone this github repo
```
git clone https://github.com/FuzhiYang/TTSR.git
cd TTSR
```
2. Download [CUFED](http://acsweb.ucsd.edu/~yuw176/event-curation.html) dataset and modify "dataset_dir" in eval.sh
3. Download pre-trained models from [google drive](https://drive.google.com/drive/folders/1CTm-r3hSbdYVCySuQ27GsrqXhhVOS-qh?usp=sharing), and modify "model_path" in eval.sh
4. Run evaluation
```
sh eval.sh
```
5. The results are in "save_dir" (default: `./eval/CUFED/TTSR`)

## Train
1. Download [CUFED](http://acsweb.ucsd.edu/~yuw176/event-curation.html) dataset and modify "dataset_dir" in train.sh
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
* yfzcopy0702@sjtu.edu.cn

