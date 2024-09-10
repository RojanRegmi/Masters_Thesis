import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# MobileNet Encoder Definition
class MobilenetEncoder(nn.Module):
    def __init__(self):
        super(MobilenetEncoder, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        mobilenet.eval()
        for param in mobilenet.parameters():
            param.requires_grad = False
        
        self.btnecks = []

        # Indices corresponding to 'conv1_relu', 'conv_dw_2_relu', 'conv_dw_4_relu', 'conv_dw_6_relu'
        output_layer_indices = [2, 4, 7, 14]
        prev_index = 0

        # Split model features into different blocks
        for idx in output_layer_indices:
            layers = list(mobilenet.features.children())[prev_index:idx + 1]
            block = nn.Sequential(*layers)
            self.btnecks.append(block)
            prev_index = idx + 1

        # Define preprocessing
        self.preprocess = lambda x: F.normalize(x * 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, block, input_tensor):
        x = self.preprocess(input_tensor)
        return self.btnecks[block](x)

# MobileNet Decoder Definition
class MobilenetDecoder(nn.Module):
    def __init__(self, padding="reflect"):
        super(MobilenetDecoder, self).__init__()
        self.btnecks = nn.ModuleList()

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
            nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=0),  
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
