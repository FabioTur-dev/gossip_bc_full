import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class NetMNIST(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, input_size=(28,28)):
        super(NetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.convs = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.conv2, nn.ReLU(), self.pool)
        self._to_linear = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def _get_conv_output(self, input_size):
        x = torch.randn(1, self.conv1.in_channels, input_size[0], input_size[1])
        x = self.convs(x)
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NetCIFAR(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, input_size=(32,32)):
        super(NetCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Per immagini 32x32, dopo 2 pooling si ottiene 8x8
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class NetSTL(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, input_size=(96,96)):
        super(NetSTL, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Per 96x96, dopo 3 pooling (divisione per 2 ad ogni step) si ottiene 12x12
        self.classifier = nn.Sequential(
            nn.Linear(256 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class NetImageNet(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, input_size=(224,224)):
        super(NetImageNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
