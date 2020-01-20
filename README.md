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
- Pytorch >= 1.3 and torchvision (https://pytorch.org/)
- You can setup the environment with Anaconda, and the `requirements.txt` file we provided:
```
conda create --name py36 python=3.6
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

### Install
- Clone this repo:
```
git clone https://github.com/hytseng0509/CrossDomainFewShot.git
cd CrossDomainFewShot
```

### Datasets
- Download 5 datasets seperately with following commands.
- Set `DATASET_NAME` to: `cars`, `cub`, `miniImagenet`, `places`, `plantae`.
```
cd filelists
python3 process.py DATASET_NAME
cd ..
```
- Refer to the instruction [here](https://github.com/wyharveychen/CloserLookFewShot#self-defined-setting) for constructing your own dataset.

### Feature encoder pre-training
- Set `METHOD` to `baseline++` for MatchineNet, `baseline` for other methods.
```
python3 train_baseline.py --method METHOD --dataset miniImageNet --name encoder --train_aug
```

### Training with multiple seen domains
- Baseline training w/o feature-wise transformations
```
python3 train_baseline.py --method METHOD --dataset multi --testset TESTSET --name METHOD_TESTSET --warmup encoder --train_aug
```
- Training w/ learning-to-learned feature-wise transformations
```
python3 train.py --method METHOD --dataset multi --testset TESTSET --name lft_METHOD_TESTSET --warmup encoder --train_aug
```
- Set `METHOD` to: `matchingnet`, `relationnet_softmax`, `gnnnet`.
- Set `TESTSET` to: `cars`, `cub`, `places`, `plantae`.

### Evaluation
- Test on the unseen domain

## Note
- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
- The dataset, model, and code are for non-commercial research purposes only.
