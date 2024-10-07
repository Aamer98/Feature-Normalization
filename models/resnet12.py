import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """
    A custom residual block used in the ResNet12 architecture.

    Args:
        ni (int): Number of input channels.
        no (int): Number of output channels.
        stride (int): Stride for the convolutional layer.
        dropout (float, optional): Dropout rate. Default is 0.
        groups (int, optional): Number of groups for group convolution. Default is 1.
    """
    def __init__(self, ni, no, stride, dropout=0, groups=1):
        super(Block, self).__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv0 = nn.Conv2d(ni, no, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(no)
        self.conv1 = nn.Conv2d(no, no, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(no)
        self.conv2 = nn.Conv2d(no, no, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(no)

        self.shortcut = nn.Sequential()
        if stride != 1 or ni != no:
            self.shortcut = nn.Conv2d(ni, no, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        """
        Forward pass of the Block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the block.
        """
        y = F.relu(self.bn0(self.conv0(x)), inplace=True)
        y = self.dropout(y)
        y = F.relu(self.bn1(self.conv1(y)), inplace=True)
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        shortcut = self.shortcut(x)
        out = F.relu(y + shortcut, inplace=True)
        return out

class Resnet12(nn.Module):
    """
    ResNet12 model for feature extraction in few-shot learning.

    Args:
        width (int): Width multiplier for the network channels.
        dropout (float): Dropout rate.
    """
    def __init__(self, width, dropout):
        super(Resnet12, self).__init__()
        self.output_size = 512
        # Widths for each group of blocks
        self.widths = [int(x * width) for x in [64, 128, 256]]
        self.widths.append(self.output_size)
        self.bn_out = nn.BatchNorm1d(self.output_size)

        start_width = 3  # Number of input channels (RGB)
        for i in range(len(self.widths)):
            stride = 1
            setattr(self, f"group_{i}", Block(start_width, self.widths[i], stride, dropout))
            start_width = self.widths[i]

    def add_classifier(self, nclasses, name="classifier"):
        """
        Adds a fully connected layer for classification.

        Args:
            nclasses (int): Number of classes.
            name (str, optional): Name of the classifier attribute. Default is "classifier".
        """
        setattr(self, name, nn.Linear(self.output_size, nclasses))

    def up_to_embedding(self, x):
        """
        Applies the residual groups up to the embedding layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after processing through the residual groups.
        """
        for i in range(len(self.widths)):
            x = getattr(self, f"group_{i}")(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

    def forward(self, x):
        """
        Forward pass of the ResNet12 model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output feature embeddings.
        """
        # Ensure input is in the correct shape
        x = x.view(-1, *x.shape[-3:])
        x = self.up_to_embedding(x)
        # Global average pooling
        x = x.mean(dim=[2, 3])
        x = F.relu(self.bn_out(x), inplace=True)
        return x
