# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch.nn as nn
import torch

__all__ = ['ResNet', 'resnet18', 'resnet20', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.

    Returns:
        nn.Conv2d: Convolutional layer with specified parameters.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, groups=groups, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.

    Returns:
        nn.Conv2d: 1x1 convolutional layer.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    """Basic residual block used in ResNet-18, ResNet-20, and ResNet-34.

    Attributes:
        expansion (int): Expansion factor for the number of channels.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride of the convolutional layer. Default is 1.
        downsample (nn.Module, optional): Downsampling layer. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width of the network. Default is 64.
        norm_layer (callable, optional): Normalization layer. Default is nn.BatchNorm2d.
        remove_last_relu (bool, optional): Whether to remove the last ReLU activation. Default is False.
    """
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, norm_layer=None, remove_last_relu=False
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

    def forward(self, x):
        """Forward pass through the BasicBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the block.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.remove_last_relu:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block used in ResNet-50, ResNet-101, and ResNet-152.

    Attributes:
        expansion (int): Expansion factor for the number of channels.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride of the convolutional layer. Default is 1.
        downsample (nn.Module, optional): Downsampling layer. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width of the network. Default is 64.
        norm_layer (callable, optional): Normalization layer. Default is nn.BatchNorm2d.
        remove_last_relu (bool, optional): Whether to remove the last ReLU activation. Default is False.
    """
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, norm_layer=None, remove_last_relu=False
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.remove_last_relu = remove_last_relu

    def forward(self, x):
        """Forward pass through the Bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the block.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.remove_last_relu:
            out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model class.

    Args:
        block_type (str): Type of block to use ('basic' or 'bottleneck').
        layers (list[int]): Number of blocks in each layer.
        zero_init_residual (bool, optional): If True, initializes the last BN in each residual branch to zero.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        width_per_group (int, optional): Base width per group. Default is 64.
        norm_layer (callable, optional): Normalization layer. Default is nn.BatchNorm2d.
        remove_last_relu (bool, optional): Whether to remove the last ReLU activation in the network. Default is False.
        input_high_res (bool, optional): If True, expects high-resolution input (uses larger kernel size and stride). Default is True.
    """

    def __init__(
        self, block_type, layers, zero_init_residual=False,
        groups=1, width_per_group=64, norm_layer=None,
        remove_last_relu=False, input_high_res=True
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        assert block_type in ['basic', 'bottleneck'], "block_type must be 'basic' or 'bottleneck'"

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        if not input_high_res:
            # For low-resolution input
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True)
            )
        else:
            # For high-resolution input
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        self.layer1 = self._make_layer(
            block_type, 64, layers[0], norm_layer=norm_layer, remove_last_relu=False
        )
        self.layer2 = self._make_layer(
            block_type, 128, layers[1], stride=2, norm_layer=norm_layer, remove_last_relu=False
        )
        self.layer3 = self._make_layer(
            block_type, 256, layers[2], stride=2, norm_layer=norm_layer, remove_last_relu=False
        )
        self.layer4 = self._make_layer(
            block_type, 512, layers[3], stride=2, norm_layer=norm_layer, remove_last_relu=remove_last_relu
        )
        self.remove_last_relu = remove_last_relu
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # This variable is added for compatibility reasons
        self.pool = self.avgpool
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # Uncomment for classification tasks

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # Initialize BatchNorm weights and biases
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # Zero-initialize BatchNorm weight
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # Zero-initialize BatchNorm weight
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block_type, planes, blocks, stride=1, norm_layer=None, remove_last_relu=False
    ):
        """Creates a ResNet layer composed of multiple blocks.

        Args:
            block_type (str): Type of block to use ('basic' or 'bottleneck').
            planes (int): Number of output channels.
            blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride for the convolutional layer.
            norm_layer (callable, optional): Normalization layer.
            remove_last_relu (bool, optional): Whether to remove the last ReLU activation.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        if block_type == 'basic':
            block = BasicBlock
        else:
            block = Bottleneck

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Downsample if necessary
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block in the layer
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, norm_layer, remove_last_relu=False
        ))
        self.inplanes = planes * block.expansion

        # Middle blocks
        for _ in range(1, blocks - 1):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, norm_layer=norm_layer, remove_last_relu=False
            ))

        # Last block
        layers.append(block(
            self.inplanes, planes, groups=self.groups,
            base_width=self.base_width, norm_layer=norm_layer, remove_last_relu=remove_last_relu
        ))

        return nn.Sequential(*layers)

    def feature_maps(self, x):
        """Extracts feature maps from the input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Feature maps after passing through the network.
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def feature(self, x):
        """Extracts feature vector from the input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Feature vector after global average pooling.
        """
        x = self.feature_maps(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output feature vector.
        """
        x = self.feature(x)
        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Returns:
        ResNet: ResNet-18 model.
    """
    model = ResNet('basic', [2, 2, 2, 2], **kwargs)
    return model


def resnet20(**kwargs):
    """Constructs a ResNet-20 model.

    Returns:
        ResNet: ResNet-20 model.
    """
    model = ResNet('basic', [2, 2, 2, 3], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Returns:
        ResNet: ResNet-34 model.
    """
    model = ResNet('basic', [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Returns:
        ResNet: ResNet-50 model.
    """
    model = ResNet('bottleneck', [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Returns:
        ResNet: ResNet-101 model.
    """
    model = ResNet('bottleneck', [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Returns:
        ResNet: ResNet-152 model.
    """
    model = ResNet('bottleneck', [3, 8, 36, 3], **kwargs)
    return model
