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
    def __init__(self, style_feats, vgg, decoder, alpha=1.0, num_style_img=1000, probability=0.5, randomize=False, rand_min=0.2, rand_max=1):
        super().__init__()
        self.vgg = vgg
        self.decoder = decoder
        self.alpha = alpha
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.downsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        self.to_tensor = transforms.ToTensor()
        self.style_features = style_feats
        self.num_styles = len(self.style_features)
        self.probability = probability
        self.randomize = randomize
        self.rand_min = rand_min
        self.rand_max = rand_max
        self.to_pil_img = transforms.ToPILImage()
    
    def randomize_alpha(self):

        alpha_values = [round(x, 1) for x in torch.arange(self.rand_min, self.rand_max, 0.1).tolist()]
        return random.choice(alpha_values)

    @torch.no_grad()
    def __call__(self, x):

        
        if self.randomize:
            self.alpha = self.randomize_alpha()

        # effective_prob = (1.0 - self.alpha)

        if torch.rand(1).item() < self.probability:

            x = self.to_tensor(x)
            x = x.to(device).unsqueeze(0)
            x = self.upsample(x)
            #x = x.unsqueeze(0)
            #x = self.upsample(x).to(device)

            idx = torch.randperm(self.num_styles, device=device)[0]
            style_image = self.style_features[idx].unsqueeze(0)
            
            stl_img = self.style_transfer(self.vgg, self.decoder, x, style_image, alpha=self.alpha)
            
            stl_img = self.downsample(stl_img).squeeze(0).cpu()

            stl_img = self.norm_style_tensor(stl_img)
            
            style_image = self.to_pil_img(stl_img)

            return style_image
        
        else:
            return x
    
    def norm_style_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()

        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        scaled_tensor = normalized_tensor * 255
        scaled_tensor = scaled_tensor.byte() # converts dtype to torch.uint8 between 0 and 255
        return scaled_tensor

    def preload_style_images(self, num_style_img):
        style_images = []
        total_images = os.listdir(self.style_dir)
        subset_imgs = total_images[0:num_style_img]
        for file in subset_imgs:
            img_path = os.path.join(self.style_dir, file)
            img = Image.open(img_path)
            tensor = self.to_tensor(img).unsqueeze(0)
            tensor = self.upsample(tensor).to(device)
            style_images.append(tensor)
        return torch.cat(style_images, dim=0).to(device)

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style, alpha=1.0):
        content_f = vgg(content)
        style_f = style
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)