# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional

class PixelNorm(nn.Module):
    """
    A custom PyTorch module for pixel-wise feature normalization.

    This layer normalizes the input tensor over the channel dimension for each pixel, enforcing unit variance
    across features. It can optionally keep track of running statistics similar to BatchNorm.

    Args:
        feature_sizes (List[int]): A list specifying the dimensions of the feature map, typically [C, H, W].
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.
        momentum (float, optional): The value used for the running magnitude computation. Defaults to 0.1.
        track_running_stats (bool, optional): If True, tracks the running magnitude statistics. Defaults to True.
        device (torch.device, optional): The device on which to create the tensors.
        dtype (torch.dtype, optional): The desired data type of the parameters.

    Attributes:
        running_magnitude (Tensor): The running average of the magnitude for normalization.
        num_batches_tracked (Tensor): The number of batches processed, used for computing running averages.

    Example:
        >>> pixel_norm = PixelNorm([64, 32, 32])
        >>> input = torch.randn(16, 64, 32, 32)
        >>> output = pixel_norm(input)
    """

    def __init__(
        self,
        feature_sizes: List[int],
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        # Initialize the running magnitude buffer
        self.register_buffer(
            'running_magnitude',
            torch.ones(*feature_sizes, **factory_kwargs)
        )
        self.running_magnitude: Optional[Tensor]

        # Initialize the batch counter
        self.register_buffer(
            'num_batches_tracked',
            torch.tensor(
                0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}
            )
        )

        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        """
        Resets the running statistics for running_magnitude and num_batches_tracked.
        """
        if self.track_running_stats:
            self.running_magnitude.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for the PixelNorm layer.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Normalized tensor with the same shape as input.
        """
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # Update the batch counter
            self.num_batches_tracked += 1

            if self.momentum is None:
                # Use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                # Use exponential moving average
                exponential_average_factor = self.momentum

            # Compute the magnitude over the batch
            magnitude = torch.sqrt((input ** 2).mean(dim=0) + self.eps)
            n = input.numel() / input.size(1)

            with torch.no_grad():
                # Update running_magnitude
                self.running_magnitude = exponential_average_factor * magnitude * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_magnitude
        else:
            magnitude = self.running_magnitude

        # Normalize the input tensor
        output = input / magnitude.unsqueeze(0)

        return output
