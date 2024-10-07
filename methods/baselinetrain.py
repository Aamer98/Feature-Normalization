# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import utils

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import time

class BaselineTrain(nn.Module):
    """
    A class for training a baseline classification model.

    This class combines a feature extractor and a classifier,
    and provides methods for training the model using a specified loss function.

    Args:
        model_func (callable): A function that returns the feature extraction model.
        num_class (int): Number of classes in the classification task.
        loss_type (str, optional): Type of loss to use ('softmax' or 'dist'). Defaults to 'softmax'.

    Attributes:
        feature (nn.Module): The feature extraction model.
        classifier (nn.Module): The classifier layer.
        loss_type (str): Type of loss used for training.
        num_class (int): Number of classes.
        loss_fn (nn.Module): Loss function (CrossEntropyLoss).
        top1 (utils.AverageMeter): Meter to track top-1 accuracy.
    """

    def __init__(self, model_func, num_class, loss_type='softmax'):
        super(BaselineTrain, self).__init__()
        self.feature = model_func()

        if loss_type == 'softmax':
            # Standard linear classifier
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            # Distance-based classifier (e.g., cosine similarity)
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        else:
            raise ValueError('Unsupported loss type')

        self.loss_type = loss_type  # 'softmax' or 'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()  # For tracking top-1 accuracy

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Classification scores (logits).
        """
        x = x.cuda()
        # Extract features
        out = self.feature.forward(x)
        # Compute classification scores
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        """
        Computes the loss given input data and labels.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
            y (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        y = y.cuda()
        # Forward pass
        scores = self.forward(x)
        # Compute top-1 accuracy
        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / y.size(0), y.size(0))
        # Compute loss
        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer, logger):
        """
        Training loop over one epoch.

        Args:
            epoch (int): Current epoch number.
            train_loader (DataLoader): DataLoader for the training set.
            optimizer (Optimizer): Optimizer for updating model parameters.
            logger (Logger): Logger for recording training metrics.

        Returns:
            dict: Dictionary of average metrics over the epoch.
        """
        print_freq = 10  # Frequency of logging
        self.train()  # Set model to training mode

        meters = utils.AverageMeterSet()  # Initialize meters for tracking metrics

        end = time.time()
        for i, (X, y) in enumerate(train_loader):
            meters.update('Data_time', time.time() - end)

            optimizer.zero_grad()
            # Forward pass
            logits = self.forward(X)
            y = y.cuda()
            # Compute loss
            loss = self.loss_fn(logits, y)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute performance metrics
            perf = utils.accuracy(logits.data, y.data, topk=(1, 5))

            # Update meters
            meters.update('Loss', loss.item(), 1)
            meters.update('top1', perf['average'][0].item(), len(X))
            meters.update('top5', perf['average'][1].item(), len(X))
            meters.update('top1_per_class', perf['per_class_average'][0].item(), 1)
            meters.update('top5_per_class', perf['per_class_average'][1].item(), 1)
            meters.update('Batch_time', time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                # Log training metrics
                logger_string = (
                    'Training Epoch: [{epoch}] Step: [{step} / {steps}] Batch Time: {meters[Batch_time]:.4f} '
                    'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
                    'Top1: {meters[top1]:.4f} Top5: {meters[top5]:.4f} '
                    'Top1_per_class: {meters[top1_per_class]:.4f} '
                    'Top5_per_class: {meters[top5_per_class]:.4f} '
                ).format(
                    epoch=epoch, step=i + 1, steps=len(train_loader), meters=meters
                )
                logger.info(logger_string)

        # Log metrics at the end of the epoch
        logger_string = (
            'Training Epoch: [{epoch}] Step: [{step}] Batch Time: {meters[Batch_time]:.4f} '
            'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
            'Top1: {meters[top1]:.4f} Top5: {meters[top5]:.4f} '
            'Top1_per_class: {meters[top1_per_class]:.4f} '
            'Top5_per_class: {meters[top5_per_class]:.4f} '
        ).format(
            epoch=epoch + 1, step=0, meters=meters
        )
        logger.info(logger_string)

        return meters.averages()

    def test_loop(self, val_loader):
        """
        Evaluation loop over the validation set.

        Args:
            val_loader (DataLoader): DataLoader for the validation set.

        Returns:
            int: Returns -1 as validation is not implemented in this method.
        """
        return -1  # No validation implemented, just save model during iteration
