# Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation 
[[Project Page]]()[[Paper]](https://openreview.net/forum?id=SJl5Np4tPr)

Pytorch implementation for our cross-domain few-shot classification method. With the proposed learned feature-wise transformation layers, we are able to:

1. improve the performance of exisiting few-shot classification methods under **cross-domain** setting
2. achieve stat-of-the-art performance under **single-domain** setting.

Contact: Hung-Yu Tseng (htseng6@ucmerced.edu)

## Paper
Please cite our paper if you find the code or dataset useful for your research.

Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation<br>
[Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/), [Hsin-Ying Lee](http://vllab.ucmerced.edu/hylee/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)<br>
International Conference on Learning Representations (ICLR), 2020 (**spotlight**)
```
@inproceedings{crossdomainfewshot,
  author = {Tseng, Hung-Yu and Lee, Hsin-Ying and Huang, Jia-Bin and Yang, Ming-Hsuan},
  booktitle = {International Conference on Learning Representations},
  title = {Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation},
  year = {2020}
}
```

## Usage

### Prerequisites
- Python >= 3.5
- Pytorch 1.3 and torchvision (https://pytorch.org/)
- Json

### Install
- Clone this repo:
```
git clone https://github.com/hytseng0509/CrossDomainFewShot.git
cd CrossDomainFewShot
```

### Datasets
- Download 5 datasets seperately with following commands.
- Specify `DATASET_NAME`: `cars`, `cub`, `miniImagenet`, `places`, `plantae`.
```
cd filelists
python3 process DATASET_NAME
cd ..
```
- Refer the [instruction](https://github.com/wyharveychen/CloserLookFewShot#self-defined-setting) for constructing your own dataset.

### Train
- under construction

### Evaluate
- under construction

## Note
- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
- The dataset, model, and code are for non-commercial research purposes only.
