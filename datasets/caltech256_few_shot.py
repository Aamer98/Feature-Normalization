# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

"""
This module provides data loading utilities for the Caltech256 dataset. It includes classes
for downloading and processing the dataset, as well as data managers and batch samplers
for training machine learning models, particularly in the context of few-shot learning.

Classes:
    Caltech256: Dataset class for the Caltech256 dataset.
    SimpleDataset: Dataset class that wraps around Caltech256 for simple data loading.
    SetDataset: Dataset class for organizing data in a way suitable for few-shot learning.
    SubDataset: Helper dataset class used within SetDataset.
    EpisodicBatchSampler: Sampler for creating batches of episodes for few-shot learning.
    TransformLoader: Loads and composes image transformations.
    DataManager: Abstract base class for data managers.
    SimpleDataManager: Data manager for simple datasets.
    SetDataManager: Data manager for episodic datasets used in few-shot learning.
"""

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from . import additional_transforms as add_transforms
from abc import abstractmethod

import os
import glob
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data

class Caltech256(data.Dataset):
    """
    `Caltech256` dataset class.

    Args:
        root (string): Root directory of the dataset where the directory
            `256_ObjectCategories` exists or will be downloaded to.
        train (bool, optional): Not used (present for compatibility).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version, e.g., `transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in the root directory. If the dataset is already downloaded, it is not
            downloaded again.

    Attributes:
        data (list): List of PIL Images.
        labels (list): List of integer labels corresponding to the images.
    """
    base_folder = '256_ObjectCategories'
    url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
    filename = "256_ObjectCategories.tar"
    tgz_md5 = '67b4f42ca05d46448c6bb8ecd2220f6d'

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()