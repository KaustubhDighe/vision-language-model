import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = BasicConvBlock(in_channels, out_channels, stride, dilation)
        self.conv2 = BasicConvBlock(out_channels, out_channels, 1, dilation)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride) if stride > 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, out_channels=1):
        super(ResNet, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Encoder
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1, dilation=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2, dilation=1)  # Stride 2 convolution
        # Decoder
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dilation):
        layers = [ResidualBlock(in_channels, out_channels, stride, dilation)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.initial_maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        
        return self.final_conv(x)
