import os
import sys

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)

if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torchvision
import torch.nn as nn
import net
from function import adaptive_instance_normalization, coral
import numpy as np
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageFile
from glob import glob

import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



encoder_rel_path = 'models/vgg_normalised.pth'
decoder_rel_path = 'models/decoder.pth'

encoder_path = os.path.join(current_dir, encoder_rel_path)
decoder_path = os.path.join(current_dir, decoder_rel_path)

class NSTTransform(transforms.Transform):

    def __init__(self, style_dir, alpha=1.0):
        super().__init__()
        self.style_dir = style_dir
        self.vgg, self.decoder = self.load_models()
        self.alpha = alpha

    def __call__(self, x):

        style_image = self.get_random_image(self.style_dir)
        transform = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        x = transform(x).unsqueeze(0)

        upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        downsample = nn.Upsample(size=(32, 32), mode='bilinear')

        x = upsample(x).to(device)
        style_image = upsample(style_image).to(device)

        with torch.no_grad():
            stl_img = self.style_transfer(self.vgg, self.decoder, x, style_image, alpha=self.alpha)
        
        stl_img = stl_img.cpu()

        stl_img = to_pil(downsample(stl_img).squeeze(0))
        
        return stl_img

    def get_random_image(self, image_dir):

        all_files = os.listdir(image_dir)

        image_files = [file for file in all_files if file.lower().endswith(('jpg', 'jpeg', 'png'))]
        random_image_file = random.choice(image_files)
        random_image_path = os.path.join(image_dir, random_image_file)

        tensor_transform = transforms.ToTensor()

        stl_img = Image.open(random_image_path)
        style_tensor = tensor_transform(stl_img).unsqueeze(0)

        style_tensor = torchvision.io.read_image(random_image_path).unsqueeze(0)

        return style_tensor


    def load_models(self):

        vgg = net.vgg
        decoder = net.decoder

        try:
            vgg.load_state_dict(torch.load(encoder_path))
        
        except Exception as e:
            print(f'Error loading VGG state dict: {e}')
            raise e

        vgg = nn.Sequential(*list(vgg.children())[:31])

        try:
            decoder.load_state_dict(torch.load(decoder_path))
        
        except Exception as e:
            print(f'Error loading decoder state dict: {e}')
            raise e
        
        vgg.to(device)
        decoder.to(device)

        decoder.eval()
        vgg.eval()

        return vgg, decoder

    def style_transfer(self, vgg, decoder, content, style, alpha=1.0,
                    interpolation_weights=None):
        assert (0.0 <= alpha <= 1.0)
        content_f = vgg(content)
        style_f = vgg(style)
        if interpolation_weights:
            _, C, H, W = content_f.size()
            feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
            base_feat = adaptive_instance_normalization(content_f, style_f)
            for i, w in enumerate(interpolation_weights):
                feat = feat + w * base_feat[i:i + 1]
            content_f = content_f[0:1]
        else:
            feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)






    
