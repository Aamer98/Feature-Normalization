# Revisiting Learnable Affines for Batch Norm in Few-Shot Transfer Learning

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Installing Dependencies](#installing-dependencies)
  - [Dataset Preparation](#dataset-preparation)
    - [MiniImageNet and CD-FSL Datasets](#miniimagenet-and-cd-fsl-datasets)
    - [ImageNet](#imagenet)
    - [Dataset Splits](#dataset-splits)
  - [Setting Dataset Paths](#setting-dataset-paths)
- [Running Experiments](#running-experiments)
  - [Baseline Experiments (Table 2)](#baseline-experiments-table-2)
  - [AdaBN Experiments (Table 2)](#adabn-experiments-table-2)
  - [ImageNet Experiments (Table 1)](#imagenet-experiments-table-1)
  - [Near-Domain Few-Shot Evaluation (Table 4)](#near-domain-few-shot-evaluation-table-4)
- [Pre-trained Models](#pre-trained-models)
- [References](#references)
- [Updates](#updates)

## Introduction

This repository provides the codebase for reproducing the experiments presented in the paper **"Revisiting Learnable Affines for Batch Norm in Few-Shot Transfer Learning"**. The paper investigates the role of learnable affine parameters in Batch Normalization layers during few-shot transfer learning scenarios. The code allows you to train models and perform fine-tuning on various datasets, such as MiniImageNet, CD-FSL datasets, and ImageNet.

## Requirements

The codebase has been tested with the following versions of packages:

- `h5py==3.1.0`
- `joypy==0.2.5`
- `matplotlib==3.4.2`
- `numpy==1.21.0`
- `pandas==1.2.3`
- `Pillow==8.4.0`
- `scikit_learn==1.0.1`
- `scipy==1.6.0`
- `seaborn==0.11.2`
- `torch==1.8.1`
- `torchvision==0.9.1`
- `tqdm==4.60.0`

To install all the required packages, run:

```bash
pip install -r requirements.txt
```

## Setup

### Installing Dependencies
Ensure all dependencies listed in `requirements.txt` are installed using the command provided above.

### Dataset Preparation
The experiments utilize the MiniImageNet, CD-FSL datasets, and ImageNet datasets. Below are instructions to prepare each dataset:

#### MiniImageNet and CD-FSL Datasets
To prepare MiniImageNet and CD-FSL datasets, follow the steps detailed in the [CD-FSL benchmark repository](https://github.com/IBM/cdfsl-benchmark).

#### ImageNet
You can download the ImageNet dataset from the [Kaggle ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).

#### Dataset Splits
All the dataset training and validation split files are located in the `datasets/split_seed_1` directory.

### Setting Dataset Paths
Set the appropriate dataset paths in the `configs.py` file.

- **Source Dataset Names:** "ImageNet", "miniImageNet"
- **Target Dataset Names:** "EuroSAT", "CropDisease", "ChestX", "ISIC"

## Running Experiments

### Baseline Experiments (Table 2)

#### Baseline BN
- **To Train:** Refer to [this link](https://github.com/MosyMosy/STARTUP/tree/main/teacher_miniImageNet).
- **To Fine-Tune:**
  ```bash
  python finetune.py --save_dir ./logs/baseline_teacher --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name}_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone
  ```
- **Pre-trained Model:** [Checkpoint](https://github.com/MosyMosy/FN_Model_Zoo/blob/main/Dictionaries/baseline_teacher/checkpoint_best.pkl)

#### Baseline FN
- **To Train:** Refer to [this link](https://github.com/MosyMosy/STARTUP/tree/main/teacher_miniImageNet_na).
- **To Fine-Tune:**
  ```bash
  python finetune.py --save_dir ./logs/baseline_na_teacher --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name}_labeled_80.csv --embedding_load_path ./logs/baseline_na_teacher/checkpoint_best.pkl --freeze_backbone
  ```
- **Pre-trained Model:** [Checkpoint](https://github.com/MosyMosy/FN_Model_Zoo/tree/main/Dictionaries/baseline_na_teacher/checkpoint_best.pkl)

### AdaBN Experiments (Table 2)

#### AdaBN BN
- **To Train:**
  ```bash
  python AdaBN.py --dir ./logs/AdaBN/{dataset_name} --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet10
  ```
- **To Fine-Tune:**
  ```bash
  python finetune.py --save_dir ./logs/AdaBN/{Target dataset name} --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name}_labeled_80.csv --embedding_load_path ./logs/AdaBN/{Target dataset name}/checkpoint_best.pkl --freeze_backbone
  ```
- **Pre-trained Model:** [Checkpoint](https://github.com/MosyMosy/FN_Model_Zoo/tree/main/Dictionaries/AdaBN/{Target_dataset_name}/checkpoint_best.pkl)

#### AdaBN FN
- **To Train:**
  ```bash
  python AdaBN_na.py --dir ./logs/AdaBN_na/{dataset_name} --base_dictionary logs/baseline_na_teacher/checkpoint_best.pkl --target_dataset $target_testset --target_subset_split datasets/split_seed_1/$target_testset_unlabeled_20.csv --bsize 256 --epochs 10 --model resnet10
  ```
- **To Fine-Tune:**
  ```bash
  python finetune.py --save_dir ./logs/AdaBN_na/{Target dataset name} --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name}_labeled_80.csv --embedding_load_path ./logs/AdaBN_na/{Target dataset name}/checkpoint_best.pkl --freeze_backbone
  ```
- **Pre-trained Model:** [Checkpoint](https://github.com/MosyMosy/FN_Model_Zoo/tree/main/Dictionaries/AdaBN_na/{Target_dataset_name}/checkpoint_best.pkl)

### ImageNet Experiments (Table 1)

#### Baseline BN (ImageNet)
- **To Train:**
  ```bash
  python ImageNet.py --dir ./logs/ImageNet/ --arch resnet18 --data ./data/ILSVRC/Data/CLS-LOC --gpu 0
  ```
- **To Fine-Tune:**
  ```bash
  python ImageNet_finetune.py --save_dir ./logs/ImageNet --target_dataset {Target dataset name} --subset_split datasets/split_seed_1/{Target dataset name}_labeled_80.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone
  ```
- **Pre-trained Model:** [Checkpoint](https://github.com/MosyMosy/FN_Model_Zoo/tree/main/Dictionaries/ImageNet/checkpoint_best.pkl)

### Near-Domain Few-Shot Evaluation (Table 4)

#### Baseline BN
- **To Fine-Tune:**
  ```bash
  python finetune.py --save_dir ./logs/eval/baseline_teacher --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/baseline_teacher/checkpoint_best.pkl --freeze_backbone
  ```
- **Pre-trained Model:** [Checkpoint](https://github.com/MosyMosy/FN_Model_Zoo/tree/main/Dictionaries/baseline_teacher/checkpoint_best.pkl)

#### AdaBN BN
- **To Adapt:**
  ```bash
  python AdaBN.py --dir ./logs/AdaBN_teacher/miniImageNet --base_dictionary logs/baseline_teacher/checkpoint_best.pkl --target_dataset ImageNet_test --target_subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --bsize 256 --epochs 10 --model resnet10
  ```
- **To Fine-Tune:**
  ```bash
  python finetune.py --save_dir ./logs/AdaBN_teacher/miniImageNet --target_dataset ImageNet_test --subset_split datasets/split_seed_1/ImageNet_val_labeled.csv --embedding_load_path ./logs/AdaBN_teacher/miniImageNet/checkpoint_best.pkl --freeze_backbone
  ```
- **Pre-trained Model:** [Checkpoint](https://github.com/MosyMosy/FN_Model_Zoo/tree/main/Dictionaries/AdaBN_teacher/miniImageNet/checkpoint_best.pkl)

## Pre-trained Models
Pre-trained models are available for each experiment, enabling easy replication and validation of results. Refer to the links provided in each experiment section to download the corresponding pre-trained models.

## References
If you find this work useful, please consider citing the paper:

```
@article{YourPaper,
  title={Revisiting Learnable Affines for Batch Norm in Few-Shot Transfer Learning},
  author={YourName, Author},
  journal={Journal of Machine Learning},
  year={2024},
  volume={X},
  pages={Y-Z}
}
```

## Updates

### 2024-06-01
- Added instructions for ImageNet training and fine-tuning.
- Improved documentation for dataset preparation.

### 2024-05-15
- Included `hps.yaml` Configuration File: Added a `hps.yaml` file to streamline the process of replicating results. The file contains all hyperparameters used in our experiments and can be found in the `config` directory.

