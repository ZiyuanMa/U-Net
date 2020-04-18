import torch
import torch.nn as nn

class RC_block(nn.Module):
    def __init__(self,channel,t=3):
        super().__init__()
        self.t = t

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r_x = self.conv(x)

        for _ in range(self.t):
            r_x = self.conv(x+r_x)

        return r_x

class RRC_block(nn.Module):
    def __init__(self, channel, t=3):
        super().__init__()

        self.RC_net = nn.Sequential(
            RC_block(channel, t=t),
            RC_block(channel, t=t),
        )

    def forward(self,x):
        
        res_x = self.RC_net(x)

        return x+res_x

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(64),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(128),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(256),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(512),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(True),
            RRC_block(64),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.trans_conv(x4)
        x = self.up_conv1(torch.cat((x, x4)))
        x = self.up_conv2(torch.cat((x, x3)))
        x = self.up_conv3(torch.cat((x, x2)))
        x = self.final_conv(torch.cat((x, x1)))
        
        return x

