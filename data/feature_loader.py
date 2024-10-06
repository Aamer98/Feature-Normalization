# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

"""
This module provides functionality for loading and processing data from HDF5 files for use in machine learning tasks.
It includes:

- `SimpleHDF5Dataset`: A dataset class that handles HDF5 data.
- `init_loader`: A function to initialize and process data from an HDF5 file.

Classes:
    SimpleHDF5Dataset: A dataset class that allows indexing into HDF5 datasets.

Functions:
    init_loader(filename): Loads data from an HDF5 file and organizes it into a dictionary keyed by class labels.
"""

import torch
import numpy as np
import h5py
import os

class SimpleHDF5Dataset:
    """
    A simple dataset class that handles data stored in an HDF5 file.

    Args:
        file_handle (h5py.File, optional): An open HDF5 file handle. If None, initializes an empty dataset.

    Attributes:
        f (h5py.File or str): The HDF5 file handle or an empty string if no file is provided.
        all_feats_dset (numpy.ndarray): The dataset containing all features.
        all_labels (numpy.ndarray): The array containing all labels.
        total (int): The total number of samples in the dataset.
    """

    def __init__(self, file_handle=None):
        if file_handle is None:
            # Initialize an empty dataset
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
        else:
            # Load data from the provided HDF5 file handle
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
            # print
