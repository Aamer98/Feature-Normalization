# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import ImageEnhance

# Dictionary mapping transformation names to their corresponding PIL ImageEnhance classes
transformtypedict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color
)


class ImageJitter(object):
    """
    Applies random jitter transformations to an image.

    The class adjusts image properties such as brightness, contrast, sharpness,
    and color by random factors within specified ranges. This is useful for data
    augmentation to make models more robust to variations in image properties.

    Args:
        transformdict (dict): A dictionary specifying the transformations to apply.
            Keys are transformation names ('Brightness', 'Contrast', 'Sharpness', 'Color'),
            and values are floats representing the maximum variation from the base value (1.0).

    Example:
        transform = ImageJitter({'Brightness': 0.4, 'Color': 0.4})
        transformed_img = transform(img)
    """

    def __init__(self, transformdict):
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict
        ]

    def __call__(self, img):
        """
        Applies the specified jitter transformations to the input image.

        Args:
            img (PIL.Image.Image): The image to be transformed.

        Returns:
            PIL.Image.Image: The transformed image with random jitter applied.
        """
        out = img
        # Generate random factors for each transformation
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            # Compute the random enhancement factor within [1 - alpha, 1 + alpha]
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1.0
            # Apply the transformation and ensure the output is in RGB mode
            out = transformer(out).enhance(r).convert('RGB')

        return out
