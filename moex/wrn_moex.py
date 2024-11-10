import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility for convolution
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

# Positional Normalization (PoNo)
def pono(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = (x.var(dim=1, keepdim=True) + epsilon).sqrt()
    normalized = (x - mean) / std
    return normalized, mean, std

def calc_mean_std(feat, eps=1e-5):
    """
    Calculate mean and standard deviation for each channel in the feature map.
    """
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def moex_instance_norm(content_feat, style_feat, epsilon=1e-5):
    """
    Perform Instance Normalization by normalizing content_feat and applying the style_feat's mean and std.
    """
    # Compute mean and std for content and style features
    content_mean, content_std = calc_mean_std(content_feat, epsilon)
    style_mean, style_std = calc_mean_std(style_feat, epsilon)

    # Normalize content feature and apply style moments
    normalized_content = (content_feat - content_mean) / content_std
    output = normalized_content * style_std + style_mean

    return output


# Basic block for Wide ResNet
class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

# Wide ResNet with PoNo and Instance Normalization (IN) for MoEx
class WideResNetWithPoNoAndIN(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate=0.3, num_classes=10, factor=1, block=WideBasic):
        super(WideResNetWithPoNoAndIN, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) // 6  # Number of blocks per layer
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        # First convolutional layer
        self.conv1 = conv3x3(3, nStages[0], stride=1)
        
        # Residual layers for 3-stage architecture
        self.layer1 = self._wide_layer(block, nStages[1], n, dropout_rate, stride=factor)  # First stage
        self.layer2 = self._wide_layer(block, nStages[2], n, dropout_rate, stride=2)  # Second stage
        self.layer3 = self._wide_layer(block, nStages[3], n, dropout_rate, stride=2)  # Third stage
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, image2=None, swap_index=None, moex_type='pono'):
        # Forward pass for image1
        out1 = self.conv1(x)
        

        # Apply PoNo (Positional Normalization) MoEx after the first layer
        if image2 is not None and moex_type == 'pono':
            out2 = self.conv1(image2)

            # Apply PoNo normalization
            out1, mean1, std1 = pono(out1)
            out2, mean2, std2 = pono(out2)

            # Exchange moments between the two images
            out1 = out1 * std2 + mean2
            out2 = out2 * std1 + mean1

        out1 = self.layer1(out1)
        out1 = self.layer2(out1)

        if image2 is not None and moex_type == 'in':
            # Forward pass for style image (image2)
            out2 = self.conv1(image2)
            out2 = self.layer1(out2)
            out2 = self.layer2(out2)

            # Apply MoEx (instance normalization on content image with style moments)
            out1 = moex_instance_norm(out1, out2)
            
        
        
        out1 = self.layer3(out1)

        # Apply Instance Normalization MoEx right after the second stage (layer2)
        

        
        out1 = F.relu(self.bn1(out1))
        out1 = F.avg_pool2d(out1, 8)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.linear(out1)

        return out1

def WideResNet_28_4_WithPoNoAndIN(num_classes, dropout_rate=0.3):
    return WideResNetWithPoNoAndIN(depth=28, widen_factor=4, dropout_rate=dropout_rate, num_classes=num_classes)
