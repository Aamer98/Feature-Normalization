# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2'
]

# URLs for pre-trained models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        dilation (int, optional): Dilation rate for dilated convolution. Default is 1.

    Returns:
        nn.Conv2d: A convolutional layer with the specified parameters.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1
) -> nn.Conv2d:
    """1x1 convolution.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class BasicBlock(nn.Module):
    """Basic residual block used in ResNet-18 and ResNet-34.

    Attributes:
        expansion (int): Expansion factor for the number of channels.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layer. Default is 1.
        downsample (nn.Module, optional): Downsampling layer. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width of the network. Default is 64.
        dilation (int, optional): Dilation rate for dilated convolution. Default is 1.
        norm_layer (callable, optional): Normalization layer. Default is nn.BatchNorm2d.
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the BasicBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
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
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """Bottleneck residual block used in ResNet-50, ResNet-101, and ResNet-152.

    Attributes:
        expansion (int): Expansion factor for the number of channels.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layer. Default is 1.
        downsample (nn.Module, optional): Downsampling layer. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width of the network. Default is 64.
        dilation (int, optional): Dilation rate for dilated convolution. Default is 1.
        norm_layer (callable, optional): Normalization layer. Default is nn.BatchNorm2d.
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, affine=False)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, affine=False)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
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
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet model class.

    Args:
        block (Type[Union[BasicBlock, Bottleneck]]): Block type to use.
        layers (List[int]): Number of blocks in each layer.
        num_classes (int, optional): Number of output classes. Default is 1000.
        zero_init_residual (bool, optional): If True, initializes the last BN in each residual branch to zero.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        width_per_group (int, optional): Base width per group. Default is 64.
        replace_stride_with_dilation (Optional[List[bool]], optional): Replace strides with dilations in layers.
        norm_layer (callable, optional): Normalization layer. Default is nn.BatchNorm2d.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # Each element in the list indicates if we should replace
            # the 2x2 stride with a dilated convolution instead.
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element list, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolution and pooling layers
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build the ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        """Creates a ResNet layer composed of multiple blocks.

        Args:
            block: Block type (BasicBlock or Bottleneck).
            planes: Number of output channels.
            blocks: Number of blocks in the layer.
            stride: Stride for the convolutional layer.
            dilate: Whether to replace stride with dilation.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        # Downsample if necessary
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, affine=False),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Implementation of the forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self._forward_impl(x)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    """Generic ResNet builder function.

    Args:
        arch (str): Architecture name.
        block: Block type.
        layers (List[int]): Number of blocks in each layer.
        pretrained (bool): If True, loads pre-trained weights.
        progress (bool): If True, displays a progress bar during download.
        **kwargs: Additional keyword arguments.

    Returns:
        ResNet: The constructed ResNet model.
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: ResNet-18 model.
    """
    return _resnet(
        'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs
    )

def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: ResNet-34 model.
    """
    return _resnet(
        'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs
    )

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: ResNet-50 model.
    """
    return _resnet(
        'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )

def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: ResNet-101 model.
    """
    return _resnet(
        'resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )

def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: ResNet-152 model.
    """
    return _resnet(
        'resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )

def resnext50_32x4d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: ResNeXt-50 32x4d model.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(
        'resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )

def resnext101_32x8d(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: ResNeXt-101 32x8d model.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(
        'resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )

def wide_resnet50_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Constructs a Wide ResNet-50-2 model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: Wide ResNet-50-2 model.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(
        'wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )

def wide_resnet101_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> ResNet:
    """Constructs a Wide ResNet-101-2 model.

    Args:
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
        progress (bool, optional): If True, displays a progress bar of the download to stderr.

    Returns:
        ResNet: Wide ResNet-101-2 model.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet(
        'wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )
