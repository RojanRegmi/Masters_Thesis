import os
import sys
import numpy as np

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

    """ A class to apply neural style transfer with the help of adaIN to datasets in the training pipeline.

    Parameters:

    style_feats: Style features extracted from the style images using adaIN Encoder
    vgg: AdaIN Encoder
    decoder: AdaIN Decoder
    alpha = Strength of style transfer [between 0 and 1]
    probability = Probability of applying style transfer [between 0 and 1]
    randomize = randomly selected strength of alpha from a given range
    rand_min = Minimum value of alpha if randomized
    rand_max = Maximum value of alpha if randomized
    upsample = Upsamples the image to a size of 224 x 224
    downsample = Downsamples the image back to 32 x 32. This is specific for CIFAR. Should be downsampled according to dataset.

     """
    def __init__(self, style_feats, encoder, decoder, alpha=1.0, num_style_img=1000, probability=0.5, randomize=False, rand_min=0.2, rand_max=1, model='vgg', skip=False):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.skip = skip

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
    
    """def randomize_alpha(self):

        alpha_values = [round(x, 1) for x in torch.arange(self.rand_min, self.rand_max, 0.1).tolist()]
        return random.choice(alpha_values)"""

    @torch.no_grad()
    def __call__(self, x):

        
        #if self.randomize:
        #    self.alpha = self.randomize_alpha()

        # effective_prob = (1.0 - self.alpha)

        if torch.rand(1).item() < self.probability:
            x = self.to_tensor(x)
            x = x.to(device).unsqueeze(0)
            x = self.upsample(x)
            #x = x.unsqueeze(0)
            #x = self.upsample(x).to(device)

            idx = torch.randperm(self.num_styles, device=device)[0]
            style_image = self.style_features[idx].unsqueeze(0)
            if self.skip:
                stl_img = self.style_transfer_skip(self.encoder, self.decoder, x, style_image)

            stl_img = self.style_transfer(self.encoder, self.decoder, x, style_image)
            
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

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style):

        if self.randomize:
            alpha = np.random.uniform(low=self.rand_min, high=self.rand_max)
        else:
            alpha = self.alpha

        content_f = vgg(content)
        style_f = style
        #print(content_f.shape, style_f.shape)
        feat = adaptive_instance_normalization(content_f, style_f)

        feat = feat * alpha + content_f * (1 - alpha)
        
        return decoder(feat)
    
    @torch.no_grad()
    def style_transfer_skip(self, vgg, decoder, content, style):

        content1, content = vgg(content)
        style, style1 = vgg(style)
        alpha = self.alpha

        feat = adaptive_instance_normalization(content, style)
        feat1 = adaptive_instance_normalization(content1, style1)

        feat = feat * alpha + content * (1-alpha)
        feat1 = feat1 * alpha + content * (1-alpha)

        return decoder(feat, feat1)



