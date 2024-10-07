# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

class MetaTemplate(nn.Module):
    """
    An abstract base class for meta-learning models in few-shot learning.

    Args:
        model_func (callable): A function that returns a feature extraction model.
        n_way (int): Number of classes (ways) in the few-shot task.
        n_support (int): Number of support samples per class.
        change_way (bool, optional): If True, allows different number of ways during training and testing. Defaults to True.

    Attributes:
        n_way (int): Number of classes (ways).
        n_support (int): Number of support samples per class.
        n_query (int): Number of query samples per class (initialized to -1 and set dynamically).
        feature (nn.Module): The feature extraction model.
        feat_dim (int): Dimensionality of the feature vector.
        change_way (bool): Flag indicating if the number of ways can change.
    """

    def __init__(self, model_func, n_way, n_support, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = -1  # Will be set dynamically based on input
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim
        self.change_way = change_way  # Allows different number of ways during training and testing

    @abstractmethod
    def set_forward(self, x, is_feature):
        """
        Abstract method for forward pass.

        Args:
            x (torch.Tensor): Input data.
            is_feature (bool): Indicates whether x is already a feature or raw input data.

        Returns:
            torch.Tensor: Model output.
        """
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        """
        Abstract method to compute loss during training.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Loss value.
        """
        pass

    def forward(self, x):
        """
        Forward pass through the feature extractor.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Extracted features.
        """
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        """
        Parses the input data into support and query features.

        Args:
            x (torch.Tensor): Input data or features.
            is_feature (bool): Indicates whether x is already a feature or raw input data.

        Returns:
            tuple: (z_support, z_query) where each is a torch.Tensor.
        """
        x = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            # Reshape input to [n_way * (n_support + n_query), ...]
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            # Extract features
            z_all = self.feature.forward(x)
            # Reshape to [n_way, n_support + n_query, feat_dim]
            z_all = z_all.view(self.n_way, self.n_support + self.n_query, -1)
        # Split into support and query sets
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        """
        Computes the number of correct predictions.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: (number of correct predictions, total number of predictions)
        """
        scores = self.set_forward(x, is_feature=False)
        y_query = np.repeat(range(self.n_way), self.n_query)

        # Get top-1 predictions
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        """
        Training loop over the dataset.

        Args:
            epoch (int): Current epoch number.
            train_loader (DataLoader): DataLoader for the training data.
            optimizer (Optimizer): Optimizer for updating model parameters.
        """
        print_freq = 10  # Frequency of printing training status
        avg_loss = 0

        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:.4f}'.format(
                    epoch, i, len(train_loader), avg_loss / float(i + 1)
                ))

    def test_loop(self, test_loader, record=None):
        """
        Evaluation loop over the test dataset.

        Args:
            test_loader (DataLoader): DataLoader for the test data.
            record (dict, optional): Dictionary to store evaluation results.

        Returns:
            float: Mean accuracy over the test dataset.
        """
        acc_all = []
        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        confidence_interval = 1.96 * acc_std / np.sqrt(iter_num)
        print('{:d} Test Acc = {:.2f}% +- {:.2f}%'.format(
            iter_num, acc_mean, confidence_interval
        ))

        return acc_mean

    def set_forward_adaptation(self, x, is_feature=True):
        """
        Performs further adaptation by training a new classifier on the support set.

        Args:
            x (torch.Tensor): Input features.
            is_feature (bool, optional): Must be True since features are fixed during adaptation.

        Returns:
            torch.Tensor: Classification scores for the query set.
        """
        assert is_feature == True, 'Features must be fixed during adaptation'
        z_support, z_query = self.parse_feature(x, is_feature)

        # Reshape support and query features
        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Create labels for support set
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        # Initialize a new linear classifier
        linear_clf = nn.Linear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        # Define optimizer and loss function
        set_optimizer = torch.optim.SGD(
            linear_clf.parameters(),
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001
        )
        loss_function = nn.CrossEntropyLoss().cuda()

        # Fine-tune the classifier
        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(
                    rand_id[i: min(i + batch_size, support_size)]
                ).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)
                loss.backward()
                set_optimizer.step()

        # Compute scores on the query set
        scores = linear_clf(z_query)
        return scores
