# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

"""
This module provides data loading utilities for image datasets, including classes for
transformations and data managers that handle dataset loading, transformations, and
batch sampling for training and evaluation.

Classes:
    TransformLoader: Loads and composes image transformations.
    DataManager: Abstract base class for data managers.
    SimpleDataManager: Data manager for simple datasets.
    SetDataManager: Data manager for episodic datasets used in few-shot learning.
"""

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod

class TransformLoader:
    """
    Loads and composes image transformations based on specified parameters.

    Args:
        image_size (int): Desired output size of the images.
        normalize_param (dict, optional): Parameters for normalization transform.
            Defaults to mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
        jitter_param (dict, optional): Parameters for jitter transform.
            Defaults to Brightness=0.4, Contrast=0.4, Color=0.4.
    """

    def __init__(self, image_size, 
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        """
        Parses and returns a single transform based on the transform type.

        Args:
            transform_type (str): The name of the transform to parse.

        Returns:
            Transform: The corresponding transform object.
        """
        if transform_type == 'ImageJitter':
            # Custom jitter transform using additional_transforms module
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Scale':
            # Scales the image to a size slightly larger than the desired size
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            # Normalizes the image using the specified mean and std
            return method(**self.normalize_param)
        else:
            # For transforms that don't require additional parameters
            return method()

    def get_composed_transform(self, aug=False):
        """
        Composes a list of transforms into a single transform.

        Args:
            aug (bool, optional): If True, includes data augmentation transforms.
                Defaults to False.

        Returns:
            Transform: A composed transform object.
        """
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale', 'CenterCrop', 'ToTensor', 'Normalize']

        # Parse each transform and compose them
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    """
    Abstract base class for data managers.

    Methods:
        get_data_loader(data_file, aug): Abstract method to get a data loader.
    """
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        """
        Abstract method to obtain a data loader.

        Args:
            data_file (str): Path to the data file.
            aug (bool): Whether to apply data augmentation.

        Returns:
            DataLoader: A PyTorch data loader object.
        """
        pass 

class SimpleDataManager(DataManager):
    """
    Data manager for simple datasets.

    Args:
        image_size (int): Desired output size of the images.
        batch_size (int): Number of samples per batch.
    """
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):
        """
        Returns a data loader for the dataset.

        Args:
            data_file (str): Path to the data file.
            aug (bool): Whether to apply data augmentation.

        Returns:
            DataLoader: A PyTorch data loader object.
        """
        # Obtain the composed transform
        transform = self.trans_loader.get_composed_transform(aug)
        # Create the dataset
        dataset = SimpleDataset(data_file, transform)
        # Data loader parameters
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)       
        # Create and return the data loader
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class SetDataManager(DataManager):
    """
    Data manager for episodic datasets used in few-shot learning.

    Args:
        image_size (int): Desired output size of the images.
        n_way (int): Number of classes per episode.
        n_support (int): Number of support samples per class.
        n_query (int): Number of query samples per class.
        n_episode (int, optional): Number of episodes per epoch. Defaults to 100.
    """
    def __init__(self, image_size, n_way, n_support, n_query, n_episode=100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):
        """
        Returns a data loader for episodic training.

        Args:
            data_file (str): Path to the data file.
            aug (bool): Whether to apply data augmentation.

        Returns:
            DataLoader: A PyTorch data loader object with episodic batching.
        """
        # Obtain the composed transform
        transform = self.trans_loader.get_composed_transform(aug)
        # Create the dataset
        dataset = SetDataset(data_file, self.batch_size, transform)
        # Create an episodic batch sampler
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        # Data loader parameters
        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)       
        # Create and return the data loader
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
