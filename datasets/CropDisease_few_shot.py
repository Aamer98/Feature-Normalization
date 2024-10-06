# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

import os
import copy

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
import configs

def construct_subset(dataset, split):
    """
    Constructs a subset of the dataset based on a provided split.

    Args:
        dataset (Dataset): The original dataset.
        split (str): Path to a CSV file containing the split information.

    Returns:
        Dataset: A subset of the original dataset containing only the samples specified in the split.
    """
    split = pd.read_csv(split)['img_path'].values
    root = dataset.root
    class_to_idx = dataset.class_to_idx

    # Create targets based on class indices
    targets = [class_to_idx[os.path.dirname(i)] for i in split]
    image_names = [os.path.join(root, j) for j in split]

    # Create a copy of the dataset and update samples and targets
    dataset_subset = copy.deepcopy(dataset)
    dataset_subset.samples = list(zip(image_names, targets))
    dataset_subset.imgs = dataset_subset.samples
    dataset_subset.targets = targets
    return dataset_subset

def identity(x):
    """Identity function used as a default target transform."""
    return x

class SimpleDataset:
    """
    A simple dataset class that wraps around the ImageFolder dataset for simple data loading.

    Args:
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.
        split (str, optional): The filename of a CSV containing a split for the data to be used.
            If None, then the full dataset is used. Defaults to None.
    """

    def __init__(self, transform, target_transform=identity, split=None):
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # Load the dataset using ImageFolder
        self.d = ImageFolder(configs.CropDisease_path, 
                             transform=self.transform,
                             target_transform=self.target_transform)

        # Apply data split if provided
        if split is not None:
            print("Using Split: ", split)
            self.d = construct_subset(self.d, split)

    def __getitem__(self, i):
        """
        Retrieves the image and target at index i.

        Args:
            i (int): Index

        Returns:
            tuple: (image, target) where image is the transformed image and target is the label.
        """
        return self.d[i]

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.d)

class SetDataset:
    """
    A dataset class that organizes data for few-shot learning by grouping images by class.

    Args:
        batch_size (int): Number of images per batch.
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        split (str, optional): The filename of a CSV containing a split for the data to be used.
            If None, then the full dataset is used. Defaults to None.
    """

    def __init__(self, batch_size, transform, split=None):
        self.d = ImageFolder(configs.CropDisease_path, transform=transform)
        self.split = split

        # Apply data split if provided
        if split is not None:
            print("Using Split: ", split)
            self.d = construct_subset(self.d, split)

        # Get the list of unique class labels
        self.cl_list = range(len(self.d.classes))

        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        # Create a DataLoader for each class
        for cl in self.cl_list:
            ind = np.where(np.array(self.d.targets) == cl)[0].tolist()
            sub_dataset = torch.utils.data.Subset(self.d, ind)
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            )

    def __getitem__(self, i):
        """
        Retrieves the next batch of data from the DataLoader corresponding to class i.

        Args:
            i (int): Index of the class DataLoader to retrieve data from.

        Returns:
            tuple: A batch of (images, targets) from the specified class DataLoader.
        """
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        """
        Returns the number of unique classes in the dataset.

        Returns:
            int: The number of unique classes.
        """
        return len(self.sub_dataloader)

class EpisodicBatchSampler(object):
    """
    Sampler that yields batches of class indices for episodic training in few-shot learning.

    Args:
        n_classes (int): Total number of classes in the dataset.
        n_way (int): Number of classes to sample in each episode.
        n_episodes (int): Number of episodes (iterations) per epoch.
    """

    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        """
        Returns the number of episodes per epoch.

        Returns:
            int: The number of episodes.
        """
        return self.n_episodes

    def __iter__(self):
        """
        Yields a list of class indices for each episode.

        Yields:
            torch.Tensor: A tensor containing indices of classes sampled for the episode.
        """
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

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
        if transform_type in ['RandomSizedCrop', 'RandomResizedCrop']:
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type in ['Scale', 'Resize']:
            # Resize the image to a size slightly larger than the desired size
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            # Normalize the image using the specified mean and std
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
            transform_list = [
                'RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip',
                'ToTensor', 'Normalize'
            ]
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        # Parse each transform and compose them
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    """
    Abstract base class for data managers.

    Methods:
        get_data_loader(aug, num_workers): Abstract method to get a data loader.
    """

    @abstractmethod
    def get_data_loader(self, aug, num_workers):
        pass

class SimpleDataManager(DataManager):
    """
    Data manager for simple datasets.

    Args:
        image_size (int): Desired output size of the images.
        batch_size (int): Number of samples per batch.
        split (str, optional): The filename of a CSV containing a split for the data to be used.
            If None, then the full dataset is used. Defaults to None.
    """

    def __init__(self, image_size, batch_size, split=None):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.split = split

    def get_data_loader(self, aug, num_workers=12):
        """
        Returns a data loader for the dataset.

        Args:
            aug (bool): Whether to apply data augmentation.
            num_workers (int, optional): Number of worker threads to use. Defaults to 12.

        Returns:
            DataLoader: A PyTorch data loader object.
        """
        # Obtain the composed transform
        transform = self.trans_loader.get_composed_transform(aug)
        # Create the dataset
        dataset = SimpleDataset(transform, split=self.split)

        # Data loader parameters
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        # Create and return the data loader
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    """
    Data manager for episodic datasets used in few-shot learning.

    Args:
        image_size (int): Desired output size of the images.
        n_way (int, optional): Number of classes per episode. Defaults to 5.
        n_support (int, optional): Number of support samples per class. Defaults to 5.
        n_query (int, optional): Number of query samples per class. Defaults to 16.
        n_episode (int, optional): Number of episodes per epoch. Defaults to 100.
        split (str, optional): The filename of a CSV containing a split for the data to be used.
            If None, then the full dataset is used. Defaults to None.
    """

    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_episode=100, split=None):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.trans_loader = TransformLoader(image_size)
        self.split = split

    def get_data_loader(self, aug, num_workers=12):
        """
        Returns a data loader for episodic training.

        Args:
            aug (bool): Whether to apply data augmentation.
            num_workers (int, optional): Number of worker threads to use. Defaults to 12.

        Returns:
            DataLoader: A PyTorch data loader object with episodic batching.
        """
        # Obtain the composed transform
        transform = self.trans_loader.get_composed_transform(aug)
        # Create the dataset
        dataset = SetDataset(self.batch_size, transform, split=self.split)
        # Create an episodic batch sampler
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        # Data loader parameters
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        # Create and return the data loader
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    # Example usage
    train_few_shot_params = dict(n_way=5, n_support=5)
    base_datamgr = SetDataManager(224, n_query=16)
    base_loader = base_datamgr.get_data_loader(aug=True)

    cnt = 1
    for i, (x, label) in enumerate(base_loader):
        if i < cnt:
            print(label)
        else:
            break
