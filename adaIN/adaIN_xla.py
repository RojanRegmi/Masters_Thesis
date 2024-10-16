import os
import sys

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)

if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch_xla
import torch_xla.core.xla_model as xm


import torch.nn as nn
import torchvision.transforms.v2 as transforms
import random
from PIL import Image
import net
from function import adaptive_instance_normalization


encoder_rel_path = 'models/vgg_normalised.pth'
decoder_rel_path = 'models/decoder.pth'
encoder_path = os.path.join(current_dir, encoder_rel_path)
decoder_path = os.path.join(current_dir, decoder_rel_path)

class NSTTransform(transforms.Transform):
    def __init__(self, style_image_list, vgg, decoder, device, alpha=1.0, probability=0.5):
        print(f"Initializing NSTTransform with device: {device}")
        super().__init__()
        self.vgg = vgg
        self.decoder = decoder
        self.alpha = alpha
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        self.to_tensor = transforms.ToTensor()
        self.style_images = style_image_list
        self.num_styles = len(self.style_images)
        self.probability = probability
        self.device = device
        print(f"NSTTransform initialized with device: {self.device}")

    @torch.no_grad()
    def __call__(self, x):
        x = self.to_tensor(x)

        if torch.rand(1).item() < self.probability:
            x = x.unsqueeze(0)
            x = self.upsample(x).to(self.device)

            idx = torch.randint(0, self.num_styles, (1,), device=self.device)[0]
            style_image = self.style_images[idx].unsqueeze(0).to(self.device)
            
            stl_img = self.style_transfer(self.vgg, self.decoder, x, style_image, alpha=self.alpha)
            xm.mark_step()
            stl_img = self.downsample(stl_img).squeeze(0).cpu()
            return stl_img
        
        else:
            return x
        
    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style, alpha=1.0):
        content_f = vgg(content)
        style_f = vgg(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)