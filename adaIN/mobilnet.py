import torch.nn as nn

class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

mobnet_decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
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
    nn.Conv2d(64, 3, (3, 3)))


class SkipEncoder(nn.Module):
    def __init__(self, encoder):
        super(SkipEncoder, self).__init__()
        self.layer1 = nn.Sequential(*list(encoder.children())[:3])
        self.layer2 = nn.Sequential(*list(encoder.children())[3:5])

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        return  x1, x2

class SkipDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.reflection = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv256 = nn.Conv2d(256, 256, (3,3))
        self.conv256_128 = nn.Conv2d(256, 128, (3, 3))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv128_64 = nn.Conv2d(128, 64, (3, 3))
        self.conv_last = nn.Conv2d(64, 3, (3, 3))
        self.relu = nn.ReLU()
        
    def forward(self, x, skip_connection):
        x = self.reflection(x)
        x = self.conv256(x)
        x = self.relu(x)
        x = self.reflection(x)
        x = self.conv256(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.reflection(x)
        x = self.conv256_128(x)
        x = self.relu(x)
        x = x + skip_connection
        x = self.upsample(x)
        x = self.reflection(x)
        x = self.conv128_64(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.reflection(x)
        x = self.conv_last(x)
        
        return x