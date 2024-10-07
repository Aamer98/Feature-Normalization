# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

def init_layer(L):
    """Initializes a layer using fan-in initialization.

    Args:
        L (nn.Module): The layer to initialize.

    Notes:
        - For Conv2d layers, weights are initialized using a normal distribution with zero mean and standard deviation sqrt(2.0 / n).
        - For BatchNorm2d layers, weights are initialized to 1 and biases to 0.
    """
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    """Module to flatten the input tensor."""
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        """Flattens the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Flattened tensor.
        """
        return x.view(x.size(0), -1)

class SimpleBlock(nn.Module):
    """A simple residual block used in ResNet.

    Attributes:
        indim (int): Number of input channels.
        outdim (int): Number of output channels.
        half_res (bool): Whether to halve the resolution (i.e., downsample).

    Args:
        indim (int): Number of input channels.
        outdim (int): Number of output channels.
        half_res (bool): If True, the block will downsample the input.
    """
    maml = False  # Default value

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.half_res = half_res

        # First convolutional layer
        self.C1 = nn.Conv2d(
            indim, outdim, kernel_size=3,
            stride=2 if half_res else 1, padding=1, bias=False
        )
        self.BN1 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.C2 = nn.Conv2d(
            outdim, outdim, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2]

        # Shortcut connection
        if indim != outdim:
            self.shortcut = nn.Conv2d(
                indim, outdim, kernel_size=1,
                stride=2 if half_res else 1, bias=False
            )
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.extend([self.shortcut, self.BNshortcut])
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        # Initialize layers
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        """Forward pass through the SimpleBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)

        out = self.C2(out)
        out = self.BN2(out)

        if self.shortcut_type == 'identity':
            short_out = x
        else:
            short_out = self.BNshortcut(self.shortcut(x))

        out += short_out
        out = self.relu2(out)
        return out

class BottleneckBlock(nn.Module):
    """A bottleneck residual block used in ResNet.

    Attributes:
        indim (int): Number of input channels.
        outdim (int): Number of output channels.
        half_res (bool): Whether to halve the resolution.

    Args:
        indim (int): Number of input channels.
        outdim (int): Number of output channels.
        half_res (bool): If True, the block will downsample the input.
    """
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.half_res = half_res

        bottleneck_dim = int(outdim / 4)

        # First convolutional layer
        self.C1 = nn.Conv2d(
            indim, bottleneck_dim, kernel_size=1, bias=False
        )
        self.BN1 = nn.BatchNorm2d(bottleneck_dim)

        # Second convolutional layer
        self.C2 = nn.Conv2d(
            bottleneck_dim, bottleneck_dim, kernel_size=3,
            stride=2 if half_res else 1, padding=1, bias=False
        )
        self.BN2 = nn.BatchNorm2d(bottleneck_dim)

        # Third convolutional layer
        self.C3 = nn.Conv2d(
            bottleneck_dim, outdim, kernel_size=1, bias=False
        )
        self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [
            self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3
        ]

        # Shortcut connection
        if indim != outdim:
            self.shortcut = nn.Conv2d(
                indim, outdim, kernel_size=1,
                stride=2 if half_res else 1, bias=False
            )
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        # Initialize layers
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        """Forward pass through the BottleneckBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.shortcut_type == 'identity':
            short_out = x
        else:
            short_out = self.shortcut(x)

        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)

        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)

        out = self.C3(out)
        out = self.BN3(out)

        out += short_out
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """ResNet model class.

    Args:
        block (nn.Module): Block type to use (SimpleBlock or BottleneckBlock).
        list_of_num_layers (list[int]): Number of layers in each stage.
        list_of_out_dims (list[int]): Number of output channels for each stage.
        flatten (bool, optional): If True, flattens the output. Default is False.

    Notes:
        - The network is constructed with 4 stages.
        - Each stage consists of multiple residual blocks.
        - The `flatten` parameter controls whether the final output is flattened for classification tasks.
    """
    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=False):
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        assert len(list_of_out_dims) == 4, 'Must provide output dimensions for four stages'

        # Initial convolutional layer
        conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                block_instance = block(indim, list_of_out_dims[i], half_res)
                trunk.append(block_instance)
                indim = list_of_out_dims[i]

        if flatten:
            # Apply adaptive average pooling and flatten
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        """Forward pass through the ResNet.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        out = self.trunk(x)
        return out

def ResNet10(flatten=True):
    """Constructs a ResNet-10 model.

    Args:
        flatten (bool, optional): If True, flattens the output. Default is True.

    Returns:
        ResNet: ResNet-10 model.
    """
    return ResNet(
        SimpleBlock,
        list_of_num_layers=[1, 1, 1, 1],
        list_of_out_dims=[64, 128, 256, 512],
        flatten=flatten
    )
