import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(True),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(True),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(True),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(True),
        )
