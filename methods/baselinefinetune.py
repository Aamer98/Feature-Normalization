# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate  # Import the MetaTemplate base class

class BaselineFinetune(MetaTemplate):
    """
    A class for baseline fine-tuning in few-shot learning.

    This class fine-tunes a linear classifier on top of the feature representations
    of the support set and then evaluates it on the query set.

    Args:
        model_func (callable): A function that returns the feature extraction model.
        n_way (int): Number of classes (ways) in the few-shot task.
        n_support (int): Number of support samples per class.
        loss_type (str, optional): Type of loss to use ('softmax' or 'dist'). Defaults to 'softmax'.

    Attributes:
        loss_type (str): Type of loss used for fine-tuning.
    """

    def __init__(self, model_func, n_way, n_support, loss_type='softmax'):
        super(BaselineFinetune, self).__init__(model_func, n_way, n_support)
        self.loss_type = loss_type

    def set_forward(self, x, is_feature=True):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input data of shape [n_way, n_support + n_query, C, H, W] or pre-extracted features.
            is_feature (bool, optional): If True, x is pre-extracted features. Defaults to True.

        Returns:
            torch.Tensor: Classification scores for the query set.
        """
        # BaselineFinetune always performs adaptation (fine-tuning)
        return self.set_forward_adaptation(x, is_feature)

    def set_forward_adaptation(self, x, is_feature=True):
        """
        Fine-tunes a linear classifier on the support set and evaluates it on the query set.

        Args:
            x (torch.Tensor): Input data of shape [n_way, n_support + n_query, C, H, W] or pre-extracted features.
            is_feature (bool, optional): If True, x is pre-extracted features. Defaults to True.

        Returns:
            torch.Tensor: Classification scores for the query set.
        """
        # Ensure that features are provided (since we are not updating the backbone)
        assert is_feature == True, 'BaselineFinetune only supports testing with pre-extracted features'

        # Parse the features into support and query sets
        z_support, z_query = self.parse_feature(x, is_feature)

        # Reshape the support and query features
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Create labels for the support set
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        # Initialize the linear classifier
        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        else:
            raise ValueError('Unsupported loss type')

        linear_clf = linear_clf.cuda()

        # Define the optimizer for fine-tuning the classifier
        set_optimizer = torch.optim.SGD(
            linear_clf.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001
        )

        # Define the loss function
        loss_function = nn.CrossEntropyLoss().cuda()

        # Fine-tune the classifier on the support set
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            # Shuffle the support set indices
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                # Select a batch of support samples
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]

                # Forward pass through the classifier
                scores = linear_clf(z_batch)

                # Compute loss and backpropagate
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        # Evaluate the classifier on the query set
        scores = linear_clf(z_query)
        return scores

    def set_forward_loss(self, x):
        """
        This method is not supported for BaselineFinetune since the backbone is not updated during fine-tuning.

        Args:
            x (torch.Tensor): Input data.

        Raises:
            ValueError: Always raises an error indicating that this method is not supported.
        """
        raise ValueError('BaselineFinetune predicts on pre-trained features and does not support backbone fine-tuning')
