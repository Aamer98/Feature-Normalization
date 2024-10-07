# This code is modified from https://github.com/jakesnell/prototypical-networks

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate  # Import the MetaTemplate base class

class ProtoNet(MetaTemplate):
    """
    A class implementing the Prototypical Network for few-shot learning.

    Prototypical Networks compute a prototype representation for each class by averaging
    the embeddings (features) of support samples. Query samples are then classified based
    on the distances to these prototypes in the embedding space.

    Args:
        model_func (callable): A function that returns the feature extraction model.
        n_way (int): Number of classes (ways) in the few-shot task.
        n_support (int): Number of support samples per class.

    Attributes:
        loss_fn (nn.Module): The loss function used for training (CrossEntropyLoss).
    """

    def __init__(self, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        """
        Computes the classification scores for the query samples.

        Args:
            x (torch.Tensor): Input data or features. If `is_feature` is False, x should be of shape
                              [n_way, n_support + n_query, C, H, W]. If `is_feature` is True, x should
                              be the pre-extracted features.
            is_feature (bool, optional): Indicates whether x is already features or raw input data.
                                         Defaults to False.

        Returns:
            torch.Tensor: Classification scores (logits) for the query samples.
        """
        # Parse features into support and query sets
        z_support, z_query = self.parse_feature(x, is_feature)

        # Ensure tensors are contiguous for memory efficiency
        z_support = z_support.contiguous()
        # Compute class prototypes by averaging the support features
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # Shape: [n_way, feat_dim]
        # Reshape query features
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)  # Shape: [n_way * n_query, feat_dim]

        # Compute distances between query features and class prototypes
        dists = euclidean_dist(z_query, z_proto)  # Shape: [n_way * n_query, n_way]
        # Convert distances to logits by negating (since lower distance means higher similarity)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        """
        Computes the loss over a batch of episodes.

        Args:
            x (torch.Tensor): Input data or features.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        # Create labels for query samples
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        # Compute classification scores
        scores = self.set_forward(x)

        # Compute loss using CrossEntropyLoss
        return self.loss_fn(scores, y_query)

def euclidean_dist(x, y):
    """
    Computes the Euclidean distance between each pair of vectors from two sets.

    Args:
        x (torch.Tensor): A tensor of shape [N, D], where N is the number of samples and D is the feature dimension.
        y (torch.Tensor): A tensor of shape [M, D], where M is the number of samples.

    Returns:
        torch.Tensor: A tensor of shape [N, M] containing the pairwise distances.
    """
    # x: [N, D]
    # y: [M, D]
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1), "Feature dimensions of x and y must be the same."

    # Expand x and y to compute pairwise distances
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    # Compute Euclidean distances
    return torch.pow(x - y, 2).sum(2)
