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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images in DTD

import sys
sys.path.append("../")
from configs import *  # Import DTD_path from configs

# Identity function used as a default target transform
identity = lambda x: x

class SimpleDataset:
    """
    A simple dataset class that wraps around the DTD dataset for simple data loading.

    Args:
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.

    Attributes:
        meta (dict): A dictionary containing 'image_names' and 'image_labels'.
    """
    def __init__(self, transform, target_transform=identity):
        # Store the transforms
        self.transform = transform
        self.target_transform = target_transform

        # Initialize metadata dictionary
        self.meta = {}

        # Initialize lists to hold image names and labels
        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        # Load the DTD dataset using ImageFolder
        d = ImageFolder(DTD_path)

        # Collect image paths and labels
        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):
        """
        Retrieves the image and target at index i.

        Args:
            i (int): Index

        Returns:
            tuple: (image, target) where image is the transformed image and target is the label.
        """
        # Apply the transform to the image
        img = self.transform(self.meta['image_names'][i])
        # Apply the target transform to the label
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.meta['image_names'])


class SetDataset:
    """
    A dataset class that organizes data for few-shot learning by grouping images by class.

    Args:
        batch_size (int): Number of images per batch.
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.

    Attributes:
        sub_meta (dict): A dictionary mapping class labels to lists of image paths.
        cl_list (list): A list of class labels.
        sub_dataloader (list): A list of DataLoaders, one for each class.
    """
    def __init__(self, batch_size, transform):
        # Initialize sub_meta for each class
        self.sub_meta = {}
        self.cl_list = range(47)  # DTD has 47 classes

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        # Load the DTD dataset using ImageFolder
        d = ImageFolder(DTD_path)

        # Organize images by class label
        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)
    
        # Parameters for the DataLoader
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use main thread only or may receive multiple batches
            pin_memory=False
        )
        
        # Create a DataLoader for each class
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
        Returns the number of unique classes in the dataset.

        Returns:
            int: The number of unique classes.
        """
        return len(self.sub_dataloader)

class SubDataset:
    """
    A helper dataset class that represents data for a single class.

    Args:
        sub_meta (list): List of image paths for the class.
        cl (int): The class label.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
            Defaults to transforms.ToTensor().
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.
    """
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        # Store the class-specific data and transformations
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        """
        Retrieves the image and target at index i within the class.

        Args:
            i (int): Index within the class.

        Returns:
            tuple: (image, target) where image is the transformed image and target is the class label.
        """
        # Apply the transform to the image
        img = self.transform(self.sub_meta[i])
        # Apply the target transform to the label
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
    """
    def __init__(self, n_classes, n_way, n_episodes):
        # Store sampling parameters
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
        Yields:
            torch.Tensor: A tensor containing indices of classes sampled for the episode.
        """
        for _ in range(self.n_episodes):
            # Randomly sample n_way classes without replacement
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
        # Store transformation parameters
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
        # Get the method from torchvision.transforms
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Scale':
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
                'RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip',
                'ToTensor', 'Normalize'
            ]
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
        get_data_loader(aug): Abstract method to get a data loader.
    """
    @abstractmethod
    def get_data_loader(self, aug):
        
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

    def get_data_loader(self, aug):
        """
        Returns a data loader for the dataset.

        Args:
            aug (bool): Whether to apply data augmentation.

        Returns:
            DataLoader: A PyTorch data loader object.
        """
        # Obtain the composed transform
        transform = self.trans_loader.get_composed_transform(aug)
        # Create the dataset
        dataset = SimpleDataset(transform)

        # Data loader parameters
        data_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
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
    """
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide=100):        
        super(SetDataManager, self).__init__()
        # Store the parameters for episodic data loading
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug):
        """
        Returns a data loader for episodic training.

        Args:
            aug (bool): Whether to apply data augmentation.

        Returns:
            DataLoader: A PyTorch data loader object with episodic batching.
        """
        # Obtain the composed transform
        transform = self.trans_loader.get_composed_transform(aug)
        # Create the dataset
        dataset = SetDataset(self.batch_size, transform)
        # Create an episodic batch sampler
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        # Data loader parameters
        data_loader_params = dict(
            batch_sampler=sampler,
            num_workers=12,
            pin_memory=True
        )
        # Create and return the data loader
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    pass
