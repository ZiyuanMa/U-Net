import torch
import torch.nn as nn

class RC_block(nn.Module):
    def __init__(self,channel,t=2):
        super().__init__()
        self.t = t

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r_x = self.conv(x)

        for _ in range(self.t):
            r_x = self.conv(x+r_x)

        return r_x

class RRC_block(nn.Module):
    def __init__(self, channel, t=2):
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
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            RRC_block(64),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            RRC_block(128),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            RRC_block(256),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            RRC_block(512),
        )

        self.trans_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            RRC_block(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )

        self.up_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            RRC_block(512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )

        self.up_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            RRC_block(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )

        self.up_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            RRC_block(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            RRC_block(64),
            nn.Conv2d(64, 1, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = self.trans_conv(x4)

        x = self.up_conv1(torch.cat((x, x4), dim=1))
        x = self.up_conv2(torch.cat((x, x3), dim=1))
        x = self.up_conv3(torch.cat((x, x2), dim=1))
        x = self.final_conv(torch.cat((x, x1), dim=1))

        x = self.sigmoid(x)
        
        return x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# class Network(nn.Module):
#     def __init__(self, in_ch=3, out_ch=1):
#         super().__init__()

#         n1 = 64
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv1 = conv_block(in_ch, filters[0])
#         self.Conv2 = conv_block(filters[0], filters[1])
#         self.Conv3 = conv_block(filters[1], filters[2])
#         self.Conv4 = conv_block(filters[2], filters[3])
#         self.Conv5 = conv_block(filters[3], filters[4])

#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_conv5 = conv_block(filters[4], filters[3])

#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])

#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])

#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])

#         self.Conv1x1 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):

#         e1 = self.Conv1(x)

#         e2 = self.Maxpool1(e1)
#         e2 = self.Conv2(e2)

#         e3 = self.Maxpool2(e2)
#         e3 = self.Conv3(e3)

#         e4 = self.Maxpool3(e3)
#         e4 = self.Conv4(e4)

#         e5 = self.Maxpool4(e4)
#         e5 = self.Conv5(e5)

#         d5 = self.Up5(e5)
#         d5 = torch.cat((e4, d5), dim=1)

#         d5 = self.Up_conv5(d5)

#         d4 = self.Up4(d5)
#         d4 = torch.cat((e3, d4), dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         d3 = torch.cat((e2, d3), dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         d2 = torch.cat((e1, d2), dim=1)
#         d2 = self.Up_conv2(d2)

#         out = self.Conv1x1(d2)

#         return torch.sigmoid(out)