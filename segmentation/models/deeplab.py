import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------
# Atrous Spatial Pyramid Pooling (ASPP)
# ---------------------------------------------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        feat1 = self.atrous_block1(x)
        feat2 = self.atrous_block6(x)
        feat3 = self.atrous_block12(x)
        feat4 = self.atrous_block18(x)
        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=(h, w), mode='bilinear', align_corners=False)

        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------
# DeepLabV3+ (ResNet-50 backbone)
# ---------------------------------------------------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=4, backbone='resnet50', pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        # --- Backbone ---
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            low_level_inplanes = 256
            high_level_inplanes = 2048
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            low_level_inplanes = 256
            high_level_inplanes = 2048
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not supported")

        # --- Encoder layers ---
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # low-level features
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # high-level features

        # --- ASPP module ---
        self.aspp = ASPP(high_level_inplanes, 256)

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        x = self.layer0(x)
        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        low_level_feat = self.decoder(low_level_feat)

        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x
