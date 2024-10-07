# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch.nn as nn

class DataParallelWrapper(nn.Module):
    """
    A wrapper class for PyTorch models that allows dynamic method forwarding in a data-parallel setting.

    Args:
        module (nn.Module): The neural network module to be wrapped.

    Methods:
        forward(mode, *args, **kwargs): Dynamically calls the specified method `mode` of the underlying module with the provided arguments.
    """

    def __init__(self, module):
        """
        Initializes the DataParallelWrapper.

        Args:
            module (nn.Module): The neural network module to be wrapped.
        """
        super(DataParallelWrapper, self).__init__()
        self.module = module

    def forward(self, mode, *args, **kwargs):
        """
        Dynamically calls the specified method of the underlying module.

        Args:
            mode (str): The name of the method to call on the underlying module.
            *args: Variable length argument list to pass to the method.
            **kwargs: Arbitrary keyword arguments to pass to the method.

        Returns:
            The result of the method call on the underlying module.
        """
        return getattr(self.module, mode)(*args, **kwargs)
