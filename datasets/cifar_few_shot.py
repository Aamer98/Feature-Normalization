# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import additional_transforms as add_transforms
from abc import abstractmethod
from torchvision.datasets import CIFAR100, CIFAR10

# Identity function for default target transformation
identity = lambda x: x

class SimpleDataset:
    """
    A simple dataset class that loads and transforms images from CIFAR100 or CIFAR10 datasets.

    Args:
        mode (str): The mode of the dataset, can be 'base', 'val', or 'novel'.
        dataset (str): The name of the dataset to use, either 'CIFAR100' or 'CIFAR10'.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.
    """

    def __init__(self, mode, dataset, transform, target_transform=identity):
        self.transform = transform
        self.dataset = dataset
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        if self.dataset == "CIFAR100":
            # Load CIFAR100 dataset
            d = CIFAR100("./", train=True, download=True)
            # Divide the dataset into 'base', 'val', and 'novel' sets based on label modulo 3
            for i, (data, label) in enumerate(d):
                if mode == "base":
                    if label % 3 == 0:
                        self.meta['image_names'].append(data)
                        self.meta['image_labels'].append(label)
                elif mode == "val":
                    if label % 3 == 1:
                        self.meta['image_names'].append(data)
                        self.meta['image_labels'].append(label)
                else:
                    if label % 3 == 2:
                        self.meta['image_names'].append(data)
                        self.meta['image_labels'].append(label)

        elif self.dataset == "CIFAR10":
            # Load CIFAR10 dataset
            d = CIFAR10("./", train=True, download=True)
            # For CIFAR10, only use 'novel' mode, include all data
            for i, (data, label) in enumerate(d):
                if mode == "novel":
                    self.meta['image_names'].append(data)
                    self.meta['image_labels'].append(label)

    def __getitem__(self, i):
        """
        Retrieves the image and target at index i.

        Args:
            i (int): Index of the item to retrieve.

        Returns:
            tuple: (image, target) where image is the transformed image and target is the transformed label.
        """
        img = self.transform(self.meta['image_names'][i])
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

    def __init__(self, mode, dataset, batch_size, transform):
        """
        Initializes the SetDataset.

        Args:
            mode (str): The mode of the dataset, can be 'base', 'val', or 'novel'.
            dataset (str): The name of the dataset to use, either 'CIFAR100' or 'CIFAR10'.
            batch_size (int): Number of images per batch.
            transform (callable): A function/transform that takes in an image and returns a transformed version.
        """
        self.sub_meta = {}
        self.cl_list = range(100)  # CIFAR100 has 100 classes
        self.dataset = dataset

        # Determine the class type based on mode
        # Classes are divided into three groups based on label modulo 3
        # type_ = 0 for 'base', type_ = 1 for 'val', type_ = 2 for 'novel'
        if mode == "base":
            type_ = 0
        elif mode == "val":
            type_ = 1
        else:
            type_ = 2

        # Initialize sub_meta for relevant classes
        for cl in self.cl_list:
            if cl % 3 == type_:
                self.sub_meta[cl] = []

        if self.dataset == "CIFAR100":
            # Load CIFAR100 dataset
            d = CIFAR100("./", train=True, download=True)
        elif self.dataset == "CIFAR10":
            # Load CIFAR10 dataset
            d = CIFAR10("./", train=True, download=True)

        # Organize images by class
        for i, (data, label) in enumerate(d):
            if label % 3 == type_:
                self.sub_meta[label].append(data)

        # Create a DataLoader for each class
        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use main thread only to avoid multiple batches
            pin_memory=False
        )
        for cl in self.cl_list:
            if cl % 3 == type_:
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
        sub_meta (list): List of images for the class.
        cl (int): The class label.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            Defaults to transforms.ToTensor().
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
            Defaults to the identity function.
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
            i (int): Index within the class.

        Returns:
            tuple: (image, target) where target is the class label.
        """
        img = self.transform(self.sub_meta[i])
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
        for i in range(self.n_episodes):
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
        pass


class SimpleDataManager(DataManager):
    """
    Data manager for simple datasets.

    Args:
        dataset (str): The name of the dataset to use, either 'CIFAR100' or 'CIFAR10'.
        image_size (int): Desired output size of the images.
        batch_size (int): Number of samples per batch.
    """

    def __init__(self, dataset, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.dataset = dataset

    def get_data_loader(self, mode, aug):
        """
        Returns a data loader for the dataset.

        Args:
            mode (str): The mode of the dataset, can be 'base', 'val', or 'novel'.
            aug (bool): Whether to apply data augmentation.

        Returns:
            DataLoader: A PyTorch data loader object.
        """
        # Obtain the composed transform
        transform = self.trans_loader.get_composed_transform(aug)
        # Create the dataset
        dataset = SimpleDataset(mode, self.dataset, transform)

        # Data loader parameters
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        # Create and return the data loader
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    """
    Data manager for episodic datasets used in few-shot learning.

    Args:
        mode (str): The mode of the dataset, can be 'base', 'val', or 'novel'.
        dataset (str): The name of the dataset to use, either 'CIFAR100' or 'CIFAR10'.
        image_size (int): Desired output size of the images.
        n_way (int, optional): Number of classes per episode. Defaults to 5.
        n_support (int, optional): Number of support samples per class. Defaults to 5.
        n_query (int, optional): Number of query samples per class. Defaults to 16.
        n_episode (int, optional): Number of episodes per epoch. Defaults to 100.
    """

    def __init__(self, mode, dataset, image_size, n_way=5, n_support=5, n_query=16, n_episode=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.mode = mode
        self.dataset = dataset

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
        dataset = SetDataset(self.mode, self.dataset, self.batch_size, transform)
        # Create an episodic batch sampler
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        # Data loader parameters
        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)
        # Create and return the data loader
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


if __name__ == '__main__':
    pass
