"""
This script implements the adaptation of a pre-trained model to a target dataset using domain adaptation techniques.

It loads a pre-trained backbone model, prepares the target dataset, and performs adaptation by updating the model's batch normalization layers.

Classes:
    - apply_twice: A wrapper class for applying transformations twice, used in SimCLR training.

Functions:
    - main(args): The main function that orchestrates the adaptation process.
    - checkpoint(model, save_path, epoch): Saves the model checkpoint.
    - load_checkpoint(model, load_path, device): Loads the model checkpoint.
    - adapt(model, trainloader, epoch, num_epochs, logger, args, device, turn_off_sync=False): Adapts the model to the target dataset.

Usage:
    Run the script with the appropriate command-line arguments to perform model adaptation.

Example:
    python script_name.py --dir ./logs/AdaBN/EuroSAT --epochs 10 --base_dictionary path_to_pretrained_model.pkl

Note:
    This script assumes that the datasets and models modules are properly defined and available in the Python path.
"""

import random
import math
import copy
from datasets import miniImageNet_few_shot, tiered_ImageNet_few_shot, ImageNet_few_shot
from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
from collections import OrderedDict
import warnings
import models
import time
import data
import utils
import sys
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets
import torch.utils.data
from configs import miniImageNet_path, ISIC_path, ChestX_path, CropDisease_path, EuroSAT_path

torch.cuda.empty_cache()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# import wandb  # Uncomment if using Weights and Biases for logging

class apply_twice:
    """
    A wrapper for torchvision transforms. The transform is applied twice for SimCLR training.

    Args:
        transform (callable): The transformation to apply to the image.
        transform2 (callable, optional): An optional second transformation. If not provided, defaults to 'transform'.
    """

    def __init__(self, transform, transform2=None):
        self.transform = transform
        self.transform2 = transform2 if transform2 is not None else transform

    def __call__(self, img):
        """
        Applies the transformations to the input image.

        Args:
            img (PIL.Image or Tensor): The input image.

        Returns:
            tuple: A tuple containing two transformed images.
        """
        return self.transform(img), self.transform2(img)

def main(args):
    """
    The main function that sets up the model, datasets, and starts the adaptation process.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    torch.cuda.empty_cache()

    # Set up directories and logging
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    logger = utils.create_logger(os.path.join(
        args.dir, time.strftime("%Y%m%d-%H%M%S") + '_checkpoint.log'), __name__)
    trainlog = utils.savelog(args.dir, 'train')
    vallog = utils.savelog(args.dir, 'val')

    # Initialize Weights and Biases (if used)
    # wandb.init(project='STARTUP',
    #            group=__file__,
    #            name=f'{__file__}_{args.dir}')
    # wandb.config.update(args)

    # Log all arguments
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # Set random seeds for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ###########################
    # Create Model
    ###########################
    if args.model == 'resnet10':
        backbone = models.ResNet10()
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    ###########################
    # Create DataLoader
    ###########################

    # Create the target dataset based on the specified dataset
    if args.target_dataset == 'ISIC':
        transform = ISIC_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = ISIC_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = ISIC_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'EuroSAT':
        transform = EuroSAT_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = EuroSAT_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = EuroSAT_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'CropDisease':
        transform = CropDisease_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = CropDisease_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = CropDisease_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'ChestX':
        transform = Chest_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = Chest_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = Chest_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'miniImageNet_test':
        transform = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = miniImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = miniImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'ImageNet_test':
        transform = ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = ImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    elif args.target_dataset == 'tiered_ImageNet_test':
        if args.image_size != 84:
            warnings.warn("Tiered ImageNet: The image size is not 84x84")
        transform = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=True)
        transform_test = tiered_ImageNet_few_shot.TransformLoader(
            args.image_size).get_composed_transform(aug=False)
        dataset = tiered_ImageNet_few_shot.SimpleDataset(
            transform, split=args.target_subset_split)
    else:
        raise ValueError('Invalid target dataset specified!')

    print("Size of target dataset:", len(dataset))
    dataset_test = copy.deepcopy(dataset)

    # Apply transformations twice for SimCLR training
    transform_twice = apply_twice(transform)
    transform_test_twice = apply_twice(transform_test, transform)

    dataset.d.transform = transform_twice
    dataset_test.d.transform = transform_test_twice

    ind = torch.randperm(len(dataset))

    # Split the target dataset into train and validation sets (90% train, 10% val)
    train_ind = ind[:int(0.9 * len(ind))]
    val_ind = ind[int(0.9 * len(ind)):]

    # For now, using the entire dataset for training
    trainset = dataset

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )

    #######################################
    starting_epoch = 0

    logger.info('Loading the pre-trained model')
    load_checkpoint(backbone, args.base_dictionary, device)

    # Adapt the model to the target dataset
    for epoch in range(starting_epoch, args.epochs):
        adapt(backbone, trainloader, epoch, args.epochs, logger, args, device)

    # Save the adapted model checkpoint
    checkpoint(backbone, os.path.join(
        args.dir, f'checkpoint_best.pkl'), args.epochs)

def checkpoint(model, save_path, epoch):
    """
    Saves the model checkpoint.

    Args:
        model (nn.Module): The model to save.
        save_path (str): The path where to save the checkpoint.
        epoch (int): The current epoch number.

    Returns:
        dict: The state dictionary saved.
    """
    state_dict = {
        'model': copy.deepcopy(model.state_dict()),
        'epoch': epoch
    }

    torch.save(state_dict, save_path)
    return state_dict

def load_checkpoint(model, load_path, device):
    """
    Loads the model checkpoint from the specified path.

    Args:
        model (nn.Module): The model to load the state into.
        load_path (str): The path to the checkpoint file.
        device (torch.device): The device to map the model to.

    Returns:
        None
    """
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.eval()

def adapt(model, trainloader, epoch, num_epochs, logger, args, device, turn_off_sync=False):
    """
    Adapts the model to the target dataset by updating the batch normalization layers.

    Args:
        model (nn.Module): The model to adapt.
        trainloader (DataLoader): DataLoader for the target dataset.
        epoch (int): The current epoch.
        num_epochs (int): Total number of epochs.
        logger (logging.Logger): Logger for logging information.
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): The device to run the model on.
        turn_off_sync (bool, optional): If True, turns off synchronization. Default is False.
    """
    model.to(device)
    model.train()

    end = time.time()
    for i, ((X1, X2), _) in enumerate(trainloader):
        with torch.no_grad():
            X1 = X1.to(device)
            X2 = X2.to(device)

            # Pass the images through the model
            f1 = model(X1)
            f2 = model(X2)

            # Optional: Uncomment to inspect batch normalization running statistics
            # for layer in model.modules():
            #     if isinstance(layer, nn.BatchNorm2d):
            #         print(layer.running_mean[2])
            #         print(layer.running_var[0])
            #         break

            if (i + 1) % args.print_freq == 0:
                logger_string = (f'Training Epoch: [{epoch}/{num_epochs}] Step: [{i + 1} / {len(trainloader)}]')
                logger.info(logger_string)
                print(logger_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STARTUP')

    # General settings
    parser.add_argument('--dir', type=str, default='./logs/AdaBN/EuroSAT',
                        help='Directory to save the checkpoints and logs')
    parser.add_argument('--bsize', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Frequency (in epochs) to save the model checkpoint')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='Frequency (in epochs) to evaluate on the validation set')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Frequency (in steps per epoch) to print training stats')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness for reproducibility')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay for the model')
    parser.add_argument('--resume_latest', action='store_true',
                        help='Resume from the latest model in args.dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')

    # Model settings
    parser.add_argument('--model', type=str, default='resnet10',
                        help='Backbone model to use (e.g., resnet10)')
    parser.add_argument('--backbone_random_init', action='store_true',
                        help='Use randomly initialized backbone instead of pre-trained')

    # Base dataset settings
    parser.add_argument('--base_dataset', type=str, default='miniImageNet',
                        help='Base dataset to use for pre-training')
    parser.add_argument('--base_path', type=str, default=miniImageNet_path,
                        help='Path to the base dataset')
    parser.add_argument('--base_split', type=str,
                        help='Split for the base dataset')
    parser.add_argument('--base_no_color_jitter', action='store_true',
                        help='Remove color jitter augmentation for ImageNet')
    parser.add_argument('--base_val_ratio', type=float, default=0.05,
                        help='Proportion of base dataset set aside for validation')

    # Validation settings
    parser.add_argument('--batch_validate', action='store_true',
                        help='Validate in batches rather than on the full dataset')

    # Target dataset settings
    parser.add_argument('--target_dataset', type=str, default='EuroSAT',
                        help='The target domain dataset')
    parser.add_argument('--target_subset_split', type=str,
                        default='datasets/split_seed_1/EuroSAT_unlabeled_20.csv',
                        help='Path to the CSV file specifying the unlabeled split for the target dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resolution of the input image')
    parser.add_argument('--base_dictionary', type=str, required=True,
                        help='Path to the pre-trained base model to adapt')

    args = parser.parse_args()
    main(args)
