# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides the `ImageJitter` class, which applies random jitter transformations
to images using the PIL `ImageEnhance` module. These transformations include adjustments
to brightness, contrast, sharpness, and color, enabling data augmentation for image processing tasks.
"""

import torch
from PIL import ImageEnhance

# Dictionary mapping transformation names to their corresponding PIL ImageEnhance classes
transformtypedict = {
    'Brightness': ImageEnhance.Brightness,
    'Contrast': ImageEnhance.Contrast,
    'Sharpness': ImageEnhance.Sharpness,
    'Color': ImageEnhance.Color
}

class ImageJitter(object):
    """
    Applies random jitter transformations to an image.

    This class modifies image properties such as brightness, contrast, sharpness,
    and color by random factors within specified ranges. It is useful for data
    augmentation to improve the robustness of models to variations in image properties.

    Args:
        transformdict (dict): A dictionary specifying the transformations to apply.
            - Keys are transformation names ('Brightness', 'Contrast', 'Sharpness', 'Color').
            - Values are floats representing the maximum variation from the base value (1.0).

    Example:
        transform = ImageJitter({'Brightness': 0.4, 'Color': 0.4})
        transformed_img = transform(img)
    """

    def __init__(self, transformdict):
        """
        Initializes the ImageJitter with the specified transformations.

        Args:
            transformdict (dict): Dictionary of transformations and their corresponding maximum adjustment factors.
        """
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict
        ]

    def __call__(self, img):
        """
        Applies the jitter transformations to the input image.

        Args:
            img (PIL.Image.Image): The image to transform.

        Returns:
            PIL.Image.Image: The transformed image with random jitter applied.
        """
        out = img
        # Generate random factors for each transformation
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            # Calculate the random enhancement factor within [1 - alpha, 1 + alpha]
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            # Apply the transformation and ensure the output is in RGB mode
            out = transformer(out).enhance(r).convert('RGB')

        return out
