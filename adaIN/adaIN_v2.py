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

class NSTTransform(transforms.Transform):
    def __init__(self, style_dir, alpha=1.0):
        super().__init__()
        self.style_dir = style_dir
        self.vgg, self.decoder = self.load_models()
        self.alpha = alpha
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        self.to_tensor = transforms.ToTensor()
        self.style_images = self.preload_style_images()
        self.num_styles = len(self.style_images)

    @torch.no_grad()
    def __call__(self, x):
        x = self.to_tensor(x).unsqueeze(0)
        x = self.upsample(x).to(device)

        idx = torch.randperm(self.num_styles, device=device)[0]
        style_image = self.style_images[idx]
        
        stl_img = self.style_transfer(self.vgg, self.decoder, x, style_image, alpha=self.alpha)
        
        stl_img = self.downsample(stl_img).squeeze(0).cpu()
        return stl_img

    def preload_style_images(self):
        style_images = []
        total_images = os.listdir(self.style_dir)
        subset_imgs = total_images[0:1000]
        for file in subset_imgs:
            img_path = os.path.join(self.style_dir, file)
            img = Image.open(img_path).convert('RGB')
            tensor = self.to_tensor(img).unsqueeze(0)
            tensor = self.upsample(tensor).to(device)
            style_images.append(tensor)
        return torch.cat(style_images, dim=0).to(device)

    def load_models(self):
        vgg = net.vgg
        decoder = net.decoder
        vgg.load_state_dict(torch.load(encoder_path))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        decoder.load_state_dict(torch.load(decoder_path))
        
        vgg.to(device).eval()
        decoder.to(device).eval()
        return vgg, decoder

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style, alpha=1.0):
        content_f = vgg(content)
        style_f = vgg(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)