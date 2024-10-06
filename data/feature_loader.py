# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import numpy as np
import h5py
import os

class SimpleHDF5Dataset:
    """
    A dataset class that handles data stored in an HDF5 file.

    Args:
        file_handle (h5py.File, optional): An open HDF5 file handle. If None, initializes an empty dataset.

    Attributes:
        f (h5py.File or str): The HDF5 file handle or an empty string if no file is provided.
        all_feats_dset (numpy.ndarray): The dataset containing all feature vectors.
        all_labels (numpy.ndarray): The array containing all labels corresponding to the features.
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
            # Uncomment for debugging
            # print('Dataset initialized.')

    def __getitem__(self, i):
        """
        Retrieves the feature vector and label at index `i`.

        Args:
            i (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing a torch.Tensor of the features and the corresponding label as an int.
        """
        return torch.Tensor(self.all_feats_dset[i, :]), int(self.all_labels[i])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return self.total

def init_loader(filename):
    """
    Initializes a data loader by loading data from an HDF5 file and organizing it by class labels.

    Args:
        filename (str): The path to the HDF5 file.

    Returns:
        dict: A dictionary where each key is a class label and each value is a list of feature vectors belonging to that class.
    """
    if os.path.isfile(filename):
        print('File %s found' % filename)
    else:
        print('File %s not found' % filename)
        return None

    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    # Extract features and labels
    feats = fileset.all_feats_dset
    labels = fileset.all_labels

    print('Feature shape:', feats.shape)
    print('Last feature vector:', feats[-1])

    # Remove any trailing zero entries in features and labels
    while np.sum(feats[-1]) == 0:
        print("Removing trailing zero entries.")
        feats = np.delete(feats, -1, axis=0)
        labels = np.delete(labels, -1, axis=0)
        
    # Get a list of unique class labels
    class_list = np.unique(labels).tolist()
    inds = range(len(labels))

    # Initialize a dictionary to hold data for each class
    cl_data_file = {cl: [] for cl in class_list}
        
    # Organize features by class label
    for ind in inds:
        cl = labels[ind]
        cl_data_file[cl].append(feats[ind])

    return cl_data_file
