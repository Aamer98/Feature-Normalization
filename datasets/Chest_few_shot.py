# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

"""
This module provides data loading utilities for a custom dataset, particularly designed
for handling medical images from the Chest X-ray dataset. It includes classes for custom
datasets, data managers, and transformation loaders that are used for training machine
learning models, especially in few-shot learning scenarios.

Classes:
    CustomDatasetFromImages: Custom dataset class that loads images and their labels from CSV files.
    SimpleDataset: Wraps around the custom dataset for simple data loading.
    SetDataset: Organizes data for few-shot learning by grouping images by class.
    EpisodicBatchSampler: Sampler that yields batches of class indices for episodic training.
    TransformLoader: Loads and composes image transformations.
    DataManager: Abstract base class for data managers.
    SimpleDataManager: Data manager for simple datasets.
    SetDataManager: Data manager for episodic datasets used in few-shot learning.

Functions:
    identity(x): Identity function used as a default target transform.
"""

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset
from abc import abstractmethod

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
import configs


def identity(x):
    """Identity function that returns the input as is."""
    return x


class CustomDatasetFromImages(Dataset):
    """
    Custom dataset class that loads images and their labels from CSV files.

    Args:
        transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.
        csv_path (str, optional): Path to the CSV file containing image paths and labels.
            Defaults to configs.ChestX_path + "/Data_Entry_2017.csv".
        image_path (str, optional): Path to the directory containing images.
            Defaults to configs.ChestX_path + "/images/".
        split (str, optional): The filename of a CSV containing a split for the data to be used.
            If None, then the full dataset is used. Defaults to None.
    """

    def __init__(self, transform, target_transform=identity,
                 csv_path=configs.ChestX_path + "/Data_Entry_2017.csv",
                 image_path=configs.ChestX_path + "/images/", split=None):
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion",
            "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"
        ]

        self.labels_maps = {
            "Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2,
            "Infiltration": 3, "Mass": 4, "Nodule": 5, "Pneumothorax": 6
        }

        self.transform = transform
        self.target_transform = target_transform

        # Read the CSV file containing image names and labels
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # Extract image names and labels
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name = []
        self.labels = []
        self.split = split

        # Filter images to include only those with specific labels
        for name, label in zip(self.image_name_all, self.labels_all):
            label = label.split("|")
            if (
                len(label) == 1 and label[0] != "No Finding" and
                label[0] != "Pneumonia" and label[0] in self.used_labels
            ):
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        self.data_len = len(self.image_name)
        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

        # Apply data split if provided
        if split is not None:
            print("Using Split: ", split)
            split = pd.read_csv(split)['img_path'].values
            # Construct the index
            ind = np.concatenate(
                [np.where(self.image_name == j)[0] for j in split])
            self.image_name = self.image_name[ind]
            self.labels = self.labels[ind]
            self.data_len = len(split)

            assert len(self.image_name) == len(split)
            assert len(self.labels) == len(split)

    def __getitem__(self, index):
        """
        Retrieves the image and target at the specified index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where image is the transformed image and target is the label.
        """
        # Get image name from the dataset
        single_image_name = self.image_name[index]

        # Open image and resize
        img_as_img = Image.open(
            self.img_path + single_image_name
        ).resize((256, 256)).convert('RGB')
        img_as_img.load()

        # Get label of the image
        single_image_label = self.labels[index]

        # Apply transforms
        img = self.transform(img_as_img)
        target = self.target_transform(single_image_label)

        return img, target

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: The total number of samples in the dataset.
        """
        return self.data_len


class SimpleDataset:
    """
    A simple dataset class that wraps around the custom dataset for simple data loading.

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
        self.d = CustomDatasetFromImages(
            transform=self.transform,
            target_transform=self.target_transform,
            split=split
        )

    def __getitem__(self, i):
        """
        Retrieves the image and target at index i.

        Args:
            i (int): Index

        Returns:
            tuple: (image, target) where image is the transformed image and target is the label.
        """
        img, target = self.d[i]
        return img, target

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
        self.transform = transform
        self.split = split
        self.d = CustomDatasetFromImages(transform=self.transform, split=split)

        # Get the list of unique class labels
        self.cl_list = sorted(np.unique(self.d.labels).tolist())

        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        # Create a DataLoader for each class
        for cl in self.cl_list:
            ind = np.where(np.array(self.d.labels) == cl)[0].tolist()
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

    def __init__(
        self, image_size,
        normalize_param=dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)
    ):
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
        n_eposide (int, optional): Number of episodes per epoch. Defaults to 100.
        split (str, optional): The filename of a CSV containing a split for the data to be used.
            If None, then the full dataset is used. Defaults to None.
    """

    def __init__(
        self, image_size, n_way=5, n_support=5, n_query=16,
        n_eposide=100, split=None
    ):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.split = split
        self.trans_loader = TransformLoader(image_size)

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
        dataset = SetDataset(self.batch_size, transform, self.split)
        # Create an episodic batch sampler
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
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
    
    base_datamgr = SetDataManager(224, n_query=16, n_support=5)
    base_loader = base_datamgr.get_data_loader(aug=True)
