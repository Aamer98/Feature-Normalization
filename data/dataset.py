# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

"""
This module provides dataset classes and a batch sampler for image data used in machine learning tasks.
It includes classes for simple datasets, set datasets for few-shot learning, and an episodic batch sampler.

Classes:
    SimpleDataset: Handles loading and transforming images from a dataset.
    SetDataset: Organizes data for few-shot learning by grouping images by class.
    SubDataset: A helper class used within SetDataset to manage data for a single class.
    EpisodicBatchSampler: Samples classes and batches for episodic training in few-shot learning.
"""

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

# Identity function for default target transformation
identity = lambda x: x

class SimpleDataset:
    """
    A simple dataset class that loads images and applies transformations.

    Args:
        data_file (str): Path to the JSON file containing image paths and labels.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.

    Attributes:
        meta (dict): A dictionary containing 'image_names' and 'image_labels'.
        transform (callable): The image transform function.
        target_transform (callable): The target transform function.
    """

    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        """
        Retrieves the image and target at index i.

        Args:
            i (int): Index of the item to retrieve.

        Returns:
            tuple: (image, target) where image is the transformed image and target is the transformed label.
        """
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: The length of the dataset.
        """
        return len(self.meta['image_names'])

class SetDataset:
    """
    A dataset class for few-shot learning that organizes data by class and provides a way to sample episodes.

    Args:
        data_file (str): Path to the JSON file containing image paths and labels.
        batch_size (int): Number of images per batch.
        transform (callable): A function/transform that takes in an image and returns a transformed version.

    Attributes:
        meta (dict): A dictionary containing 'image_names' and 'image_labels'.
        cl_list (list): A list of unique class labels.
        sub_meta (dict): A dictionary mapping each class label to a list of image paths.
        sub_dataloader (list): A list of DataLoaders, one for each class.
    """

    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        # Get the list of unique class labels
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        # Organize image paths by class
        self.sub_meta = {cl: [] for cl in self.cl_list}
        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        # Create a DataLoader for each class
        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use main thread only to avoid multiple batches
            pin_memory=False
        )
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

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
        Returns the number of classes.

        Returns:
            int: The number of unique classes in the dataset.
        """
        return len(self.cl_list)

class SubDataset:
    """
    A helper dataset class that represents data for a single class.

    Args:
        sub_meta (list): A list of image paths for the class.
        cl (int): The class label.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            Defaults to transforms.ToTensor().
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.

    Attributes:
        sub_meta (list): The list of image paths for the class.
        cl (int): The class label.
        transform (callable): The image transform function.
        target_transform (callable): The target transform function.
    """

    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        """
        Retrieves the image and target at index i within the class.

        Args:
            i (int): Index of the item within the class.

        Returns:
            tuple: (image, target) where image is the transformed image and target is the class label.
        """
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        """
        Returns the number of images in the class.

        Returns:
            int: The length of the sub-dataset for the class.
        """
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    """
    Sampler that yields batches of class indices for episodic training in few-shot learning.

    Args:
        n_classes (int): Total number of classes in the dataset.
        n_way (int): Number of classes to sample in each episode.
        n_episodes (int): Number of episodes (iterations) per epoch.

    Attributes:
        n_classes (int): Total number of classes.
        n_way (int): Number of classes per episode.
        n_episodes (int): Number of episodes per epoch.
    """

    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        """
        Returns the number of episodes.

        Returns:
            int: The number of episodes per epoch.
        """
        return self.n_episodes

    def __iter__(self):
        """
        Yields a list of class indices for each episode.

        Yields:
            torch.Tensor: A tensor containing indices of classes sampled for the episode.
        """
        for _ in range(self.n_episodes):
            # Randomly sample n_way classes without replacement
            yield torch.randperm(self.n_classes)[:self.n_way]
