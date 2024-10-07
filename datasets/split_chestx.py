# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images

from shutil import copyfile
import sys
sys.path.append("../")

def identity(x):
    """Identity function used as a default target transform."""
    return x

class CustomDatasetFromImages(Dataset):
    """
    A custom dataset class for loading images and labels from the NIH Chest X-ray dataset.

    Args:
        csv_path (str, optional): Path to the CSV file containing image names and labels.
            Defaults to "Data_Entry_2017.csv".
        image_path (str, optional): Path to the folder where images are stored.
            Defaults to "/images/".
        split (str, optional): The filename of a CSV containing a split for the data to be used.
            If None, then the full dataset is used. Defaults to "ChestX_unlabeled_20.csv".

    Attributes:
        img_path (str): Path to the image folder.
        csv_path (str): Path to the CSV file.
        used_labels (list): List of labels to be used from the dataset.
        labels_maps (dict): Mapping from label names to numerical indices.
        image_name_all (np.ndarray): Array of all image names from the CSV file.
        labels_all (np.ndarray): Array of all labels from the CSV file.
        image_name (np.ndarray): Array of filtered image names to be used.
        labels (np.ndarray): Array of corresponding labels for the filtered images.
        data_len (int): Total number of samples in the filtered dataset.
        split (str): Path to the split CSV file.
    """

    def __init__(
        self,
        csv_path="Data_Entry_2017.csv",
        image_path="/images/",
        split="ChestX_unlabeled_20.csv"
    ):
        # Initialize paths and label information
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax"
        ]
        self.labels_maps = {
            "Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2,
            "Infiltration": 3, "Mass": 4, "Nodule": 5, "Pneumothorax": 6
        }

        # Read the CSV file containing image names and labels
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # Extract image names and labels from the CSV
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name = []
        self.labels = []
        self.split = split

        # Filter images and labels based on specified conditions
        for name, label in zip(self.image_name_all, self.labels_all):
            label = label.split("|")

            # Include images with a single label that is in used_labels
            if (
                len(label) == 1 and
                label[0] != "No Finding" and
                label[0] != "Pneumonia" and
                label[0] in self.used_labels
            ):
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        # Update dataset length and convert lists to numpy arrays
        self.data_len = len(self.image_name)
        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

        # Apply data split if provided
        if split is not None:
            print("Using Split: ", split)
            split = pd.read_csv(split)['img_path'].values
            # Construct indices of images to include in the split
            ind = np.concatenate(
                [np.where(self.image_name == j)[0] for j in split]
            )
            self.image_name = self.image_name[ind]
            self.labels = self.labels[ind]
            self.data_len = len(split)

            # Ensure that the lengths match
            assert len(self.image_name) == len(split)
            assert len(self.labels) == len(split)

    def __getitem__(self, index):
        """
        Retrieves the image name and label at the specified index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: (image_name, label) where `image_name` is the filename of the image
                   and `label` is the corresponding class label.
        """
        # Get image name and label at the specified index
        single_image_name = self.image_name[index]
        single_image_label = self.labels[index]

        return single_image_name, single_image_label

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.data_len

# Instantiate the dataset
ds = CustomDatasetFromImages()

# Create a new text file to store image names and labels
with open('result.txt', 'w') as sample_file:
    # Iterate over the dataset
    for i in range(len(ds)):
        sample = ds[i]
        # Write image name and label to the text file
        sample_file.write('{0},{1}\n'.format(sample[0], sample[1]))
        # Copy the image to a new directory
        copyfile('./images/' + sample[0], './ChestX8_lf/images/' + sample[0])
