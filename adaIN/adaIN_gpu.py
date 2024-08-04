import os
import sys

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)

if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import os
import random
from PIL import Image
import net
from function import adaptive_instance_normalization

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder_rel_path = 'models/vgg_normalised.pth'
decoder_rel_path = 'models/decoder.pth'
encoder_path = os.path.join(current_dir, encoder_rel_path)
decoder_path = os.path.join(current_dir, decoder_rel_path)

class NSTTransform(nn.Module):
    def __init__(self, style_feats, vgg, decoder, alpha=1.0, probability=0.5):
        super().__init__()
        self.vgg = vgg
        self.decoder = decoder
        self.alpha = alpha
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        self.style_features = style_feats
        self.num_styles = len(self.style_features)
        self.probability = probability

    @torch.no_grad()
    def forward(self, x):
        # Assume x is already a batch of tensors on GPU
        batch_size = x.size(0)
        
        # Create a mask for which images in the batch to apply style transfer
        mask = torch.rand(batch_size, device=x.device) < self.probability
        
        if mask.any():
            # Apply style transfer only to selected images
            x_style = x[mask]
            x_style = self.upsample(x_style)
            
            # Randomly select styles for each image
            style_indices = torch.randint(self.num_styles, (mask.sum(),), device=x.device)
            style_images = self.style_features[style_indices]
            
            stl_img = self.style_transfer(self.vgg, self.decoder, x_style, style_images, alpha=self.alpha)
            stl_img = self.downsample(stl_img)
            
            # Replace the styled images in the original batch
            x[mask] = stl_img

        return x

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style, alpha=1.0):
        content_f = vgg(content)
        style_f = style
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)