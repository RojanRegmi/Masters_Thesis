import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

"""decoder = nn.Sequential(
        # decoder to use MobileNetV2 until 14th layer
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(96, 128, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 32, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 16, (3, 3)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 16, (3, 3)),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 3, (3, 3))
        )"""

# Define the decoder (adjust input channels to match encoder output)
decoder = nn.Sequential(
    # decoder to use MobileNetV2 until 18th layer
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(320, 256, (3, 3)),  # Adjusted input channels from 512 to 320
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# Load MobileNetV2 and extract features
mobilenet_v2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
mobilenet_v2.eval()
encoder = mobilenet_v2.features

# Define the Net class with the MobileNetV2 encoder
class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        self.enc_layers = list(encoder.children())
        # Define encoder stages corresponding to different layers
        self.enc_1 = nn.Sequential(*self.enc_layers[:2])
        self.enc_2 = nn.Sequential(*self.enc_layers[2:4])
        self.enc_3 = nn.Sequential(*self.enc_layers[4:10])
        self.enc_4 = nn.Sequential(*self.enc_layers[10:18])

        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # Freeze encoder parameters
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = []
        x = input
        x = self.enc_1(x)
        results.append(x)
        x = self.enc_2(x)
        results.append(x)
        x = self.enc_3(x)
        results.append(x)
        x = self.enc_4(x)
        results.append(x)
        return results

    def encode(self, input):
        x = input
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)
        return x

    def calc_content_loss(self, input, target):
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
