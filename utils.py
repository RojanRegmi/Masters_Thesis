import torchvision
from torchvision.transforms.v2 import Transform
from torchvision import transforms
import torch
from adaIN.net import vgg

from typing import List, Dict

import os
import sys
import random

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)
vgg_weights_path = os.path.join(current_dir, 'adaIN/models/vgg_normalised.pth')

if module_path not in sys.path:
    sys.path.append(module_path)

def downsample_image(image, size=(32, 32)):
    """
    Downsample the given PIL image to the specified size using nn.Upsample.
    """
    # Convert the PIL image to a tensor
    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image).unsqueeze(0)  # Add batch dimension

    # Define the upsample transformation
    upsample = torch.nn.Upsample(size=size, mode='bilinear', align_corners=True)

    # Apply the upsample transformation
    downsampled_tensor = upsample(image_tensor)

    # Convert back to PIL image
    transform_to_pil = transforms.ToPILImage()
    downsampled_image = transform_to_pil(downsampled_tensor.squeeze(0))  # Remove batch dimension

    return downsampled_image

class RandomChoiceTransforms:

    def __init__(self, transforms: List[Transform], p:List[float]):
        assert len(transforms) == len(p), "The number of transforms and probabilities must match"

        self.transforms = transforms
        self.p = p 

    def __call__(self, x):

        choice = random.choices(self.transforms, weights=self.p)[0]
        return choice(x)
    

