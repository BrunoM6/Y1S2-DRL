import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
class DeepfakeCNN(nn.Module):
    def __init__(self, embedding_dim=512):
        # 512x512 -> 256x256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 128x128

        # Residual Layers
        self.layer1 = ResidualBlock(64, 128, stride=2)   # -> 64x64
        self.layer2 = ResidualBlock(128, 256, stride=2)  # -> 32x32
        self.layer3 = ResidualBlock(256, 512, stride=2)  # -> 16x16
        
        # Global Context
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Final Embedding Layer
        self.fc = nn.Linear(512, embedding_dim)
        self.bn_feat = nn.BatchNorm1d(embedding_dim) # Helps normalize embeddings

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.bn_feat(x)