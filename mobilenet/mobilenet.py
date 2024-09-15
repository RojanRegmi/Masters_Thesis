import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import models, transforms

class MobilenetEncoder(nn.Module):
    def __init__(self):
        super(MobilenetEncoder, self).__init__()
        
        # Load pre-trained MobileNetV2 model
        mobilenet = models.mobilenet_v2(pretrained=True)
        mobilenet.eval()

        # Freeze all the parameters
        for param in mobilenet.parameters():
            param.requires_grad = False
        
        # Initialize bottleneck layers as a ModuleList
        self.btnecks = nn.ModuleList()

        # Indices corresponding to the layers: 'conv1_relu', 'conv_dw_2_relu', 'conv_dw_4_relu', 'conv_dw_6_relu'
        output_layer_indices = [2, 4, 7, 14]
        prev_index = 0

        # Split model features into different blocks
        for idx in output_layer_indices:
            layers = list(mobilenet.features.children())[prev_index:idx + 1]
            block = nn.Sequential(*layers)
            self.btnecks.append(block)
            prev_index = idx + 1

        # Define preprocessing using torchvision.transforms
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

    def forward(self, block, input_tensor):
        # Apply preprocessing
        x = self.preprocess(input_tensor)
        # Apply the specific block of layers
        return self.btnecks[block](x)



# MobileNet Decoder Definition
class MobilenetDecoder(nn.Module):
    def __init__(self, padding="reflect"):
        super(MobilenetDecoder, self).__init__()
        self.btnecks = nn.ModuleList()

        # Define the number of channels for each bottleneck layer in the decoder
        new_layers = {
            32: 64,    # Example: Bottleneck layer with 32 input channels has 64 output channels
            64: 128,   # Example: Bottleneck layer with 64 input channels has 128 output channels
            128: 256,  # Example: Bottleneck layer with 128 input channels has 256 output channels
            256: 512   # Example: Bottleneck layer with 256 input channels has 512 output channels
        }

        n1 = new_layers[32]
        self.btnecks.append(self.create_btneck1(3, n1, padding_type=padding))
        
        n2 = new_layers[64]
        self.btnecks.append(self.create_btneck3(n1, n2, padding_type=padding))
        
        n3 = new_layers[128]
        self.btnecks.append(self.create_btneck3(n2, n3, padding_type=padding))
        
        n4 = new_layers[256]
        self.btnecks.append(self.create_btneck3(n3, n4, padding_type=padding))

    def create_btneck1(self, in_channels, out_channels, padding_type='reflect'):
        layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
        )
        return layers

    def create_btneck3(self, in_channels, out_channels, padding_type='reflect'):
        layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            nn.ReflectionPad2d(1),  
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),  
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),  
            nn.ReLU(inplace=True),
        )
        return layers

    def forward(self, block, input_tensor, skip=None):
        if skip is not None:
            skip_resized = F.interpolate(skip, size=(input_tensor.shape[2] * 2, input_tensor.shape[3] * 2), mode='bilinear', align_corners=True)
            input_tensor = input_tensor + skip_resized  

        bt_out = self.btnecks[block](input_tensor)

        return bt_out

 def forward(self, input_tensor):
        x = input_tensor
        for block in reversed(self.btnecks):
            x = block(x)
        return x