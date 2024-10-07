import torch
import torch.nn as nn
import copy
import warnings

class RunningEnsemble(nn.Module):
    """
    A module that maintains a running ensemble of models by averaging their parameters.
    Useful for techniques like Stochastic Weight Averaging (SWA).

    Attributes:
        model (nn.Module): The ensemble model with averaged parameters.
        num_models (torch.Tensor): The number of models that have been averaged.
        bn_updated (bool): Flag indicating whether batch norm statistics have been updated.
    """
    def __init__(self, model):
        """
        Initializes the RunningEnsemble module.

        Args:
            model (nn.Module): The initial model to start the ensemble from.
        """
        super(RunningEnsemble, self).__init__()
        self.model = copy.deepcopy(model)
        self.model.eval()

        # Freeze parameters of the ensemble model
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Register a buffer to keep track of the number of models
        self.register_buffer('num_models', torch.zeros(1))
        self.bn_updated = False

    def update(self, model):
        """
        Updates the ensemble model by averaging the parameters with a new model.

        Args:
            model (nn.Module): The new model to incorporate into the ensemble.
        """
        alpha = 1 / (self.num_models + 1)
        for p_ensemble, p_new in zip(self.model.parameters(), model.parameters()):
            p_ensemble.data *= (1 - alpha)
            p_ensemble.data += p_new.data * alpha

        self.num_models += 1
        self.bn_updated = False  # BatchNorm statistics need to be updated

    @staticmethod
    def _reset_bn(module):
        """
        Resets the running mean and variance of batch normalization layers.

        Args:
            module (nn.Module): Module to reset.
        """
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    @staticmethod
    def _get_momenta(module, momenta):
        """
        Stores the current momentum of batch normalization layers.

        Args:
            module (nn.Module): Module from which to get momentum.
            momenta (dict): Dictionary to store momenta.
        """
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum

    @staticmethod
    def _set_momenta(module, momenta):
        """
        Restores the momenta of batch normalization layers.

        Args:
            module (nn.Module): Module to set momentum.
            momenta (dict): Dictionary containing stored momenta.
        """
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def update_bn(self, loader):
        """
        Updates the batch normalization statistics by running the ensemble model
        over the data loader.

        Args:
            loader (DataLoader): DataLoader providing the data over which to update BN stats.
        """
        self.model.train()
        self.model.apply(self._reset_bn)
        is_cuda = next(self.model.parameters()).is_cuda

        momenta = {}
        self.model.apply(lambda module: self._get_momenta(module, momenta))
        n = 0  # Number of samples processed

        for X, _ in loader:
            if is_cuda:
                X = X.cuda()

            b = X.size(0)
            momentum = b / float(n + b)

            for module in momenta.keys():
                module.momentum = momentum

            self.model(X)
            n += b

        self.model.apply(lambda module: self._set_momenta(module, momenta))
        self.model.eval()
        self.bn_updated = True

    def forward(self, x):
        """
        Forward pass through the ensemble model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the ensemble model.
        """
        if not self.bn_updated:
            warnings.warn('BatchNorm running mean and variance are not updated! Use with care.')
        return self.model(x)
