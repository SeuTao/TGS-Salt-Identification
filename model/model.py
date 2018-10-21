import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from ibnnet import resnext101_ibn_a,IBN
from senet import se_resnext50_32x4d

def Stconv3x3(in_, out, bias=True):
    return nn.Conv2d(in_, out, 3, padding=1, bias=bias)

def Stconv7x7(in_, out, bias=True):
    return nn.Conv2d(in_, out, 7, padding=3, bias=bias)

def Stconv5x5(in_, out, bias=True):
    return nn.Conv2d(in_, out, 5, padding=2, bias=bias)

def Stconv1x1(in_, out, bias=True):
    return nn.Conv2d(in_, out, 1, padding=0, bias=bias)

class StConvRelu(nn.Module):
    def __init__(self, in_, out, kernel_size, norm_type = None):
        super(StConvRelu,self).__init__()

        is_bias = True
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            self.norm = nn.BatchNorm2d(out)
            is_bias = False

        elif norm_type == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out)
            is_bias = True

        if kernel_size == 3:
            self.conv = Stconv3x3(in_, out, is_bias)
        elif kernel_size == 7:
            self.conv = Stconv7x7(in_, out, is_bias)
        elif kernel_size == 5:
            self.conv = Stconv5x5(in_, out, is_bias)
        elif kernel_size == 1:
            self.conv = Stconv1x1(in_, out, is_bias)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.norm_type is not None:
            x = self.conv(x)
            x = self.norm(x)
            x = self.activation(x)
        else:
            x = self.conv(x)
            x = self.activation(x)
        return x

class ImprovedIBNaDecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(ImprovedIBNaDecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = IBN(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class Decoder(nn.Module):
    def __init__(self,in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(channels, out_channels, kernel_size=3,padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
        self.SCSE = SCSEBlock(out_channels)

    def forward(self, x, e = None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e],1)
            x = F.dropout2d(x, p = 0.50)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.SCSE(x)
        return x

class model34_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1, mask_class = 2):
        super(model34_DeepSupervion, self).__init__()

        self.num_classes = num_classes

        self.encoder = torchvision.models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, mask_class)

        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.decoder5 = Decoder(256 + 512, 512, 64)
        self.decoder4 = Decoder(64 + 256, 256, 64)
        self.decoder3 = Decoder(64 + 128, 128, 64)
        self.decoder2 = Decoder(64 + 64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2,mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')),1)
        hypercol = F.dropout2d(hypercol, p = 0.50)

        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=128,mode='bilinear')),1)

        x_final = self.logits_final( hypercol_add_center)
        return center_fc, x_no_empty, x_final

class model50A_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1):
        super(model50A_DeepSupervion, self).__init__()

        self.num_classes = num_classes
        self.encoder = se_resnext50_32x4d()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(512*4, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)

        self.center = nn.Sequential(nn.Conv2d(512*4, 512, kernel_size=3,padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2,stride=2))

        self.decoder5 = Decoder(256 + 512*4, 512, 64)
        self.decoder4 = Decoder(64 + 256*4, 256, 64)
        self.decoder3 = Decoder(64 + 128*4, 128, 64)
        self.decoder2 = Decoder(64 + 64*4, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.logits_no_empty = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.logits_final = nn.Sequential(nn.Conv2d(320+64, 64, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/4
        conv2 = self.conv2(conv1) #1/4
        conv3 = self.conv3(conv2) #1/8
        conv4 = self.conv4(conv3) #1/16
        conv5 = self.conv5(conv4) #1/32

        center_512 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_512)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        f = self.center(conv5)
        d5 = self.decoder5(f, conv5)
        d4 = self.decoder4(d5, conv4)
        d3 = self.decoder3(d4, conv3)
        d2 = self.decoder2(d3, conv2)
        d1 = self.decoder1(d2)

        hypercol = torch.cat((
            d1,
            F.upsample(d2, scale_factor=2,mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear')),1)
        hypercol = F.dropout2d(hypercol, p = 0.50)

        x_no_empty = self.logits_no_empty(hypercol)
        hypercol_add_center = torch.cat((
            hypercol,
            F.upsample(center_64, scale_factor=128,mode='bilinear')),1)

        x_final = self.logits_final( hypercol_add_center)
        return center_fc, x_no_empty, x_final

class model101A_DeepSupervion(nn.Module):
    def __init__(self, num_classes=1, ):
        super(model101A_DeepSupervion, self).__init__()

        self.num_classes = num_classes
        num_filters = 32
        baseWidth = 4
        cardinality = 32
        self.encoder = resnext101_ibn_a(baseWidth, cardinality, pretrained = True)


        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center_global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.center_conv1x1 = nn.Conv2d(2048, 64, kernel_size=1)
        self.center_fc = nn.Linear(64, 2)

        self.center_se = SELayer(512*4)
        self.center = ImprovedIBNaDecoderBlock(512*4,  num_filters * 8)

        self.dec5_se = SELayer(512*4 + num_filters * 8)
        self.dec5 = ImprovedIBNaDecoderBlock(512*4 + num_filters * 8, num_filters * 8)

        self.dec4_se = SELayer(256*4 + num_filters * 8)
        self.dec4 = ImprovedIBNaDecoderBlock(256*4 + num_filters * 8, num_filters * 8)

        self.dec3_se = SELayer(128*4 + num_filters * 8)
        self.dec3 = ImprovedIBNaDecoderBlock(128*4 + num_filters * 8, num_filters * 4)

        self.dec2_se = SELayer(64*4 + num_filters * 4)
        self.dec2 = ImprovedIBNaDecoderBlock(64*4 + num_filters * 4, num_filters * 4)

        self.logits_no_empty = nn.Sequential(StConvRelu(num_filters * 4, num_filters, 3),
                                             nn.Dropout2d(0.5),
                                             nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))


        self.logits_final = nn.Sequential(StConvRelu(num_filters * 4 + 64, num_filters, 3),
                                          nn.Dropout2d(0.5),
                                          nn.Conv2d(num_filters, 1, kernel_size=1, padding=0))

    def forward(self, x):
        conv1 = self.conv1(x)     #1/2
        conv2 = self.conv2(conv1) #1/2
        conv3 = self.conv3(conv2) #1/4
        conv4 = self.conv4(conv3) #1/8
        conv5 = self.conv5(conv4) #1/16

        center_2048 = self.center_global_pool(conv5)
        center_64 = self.center_conv1x1(center_2048)
        center_64_flatten = center_64.view(center_64.size(0), -1)
        center_fc = self.center_fc(center_64_flatten)

        center = self.center(self.center_se(self.pool(conv5)))#1/16

        dec5 = self.dec5(self.dec5_se(torch.cat([center, conv5], 1)))#1/8
        dec4 = self.dec4(self.dec4_se(torch.cat([dec5, conv4], 1)))  #1/4
        dec3 = self.dec3(self.dec3_se(torch.cat([dec4, conv3], 1)))  #1/2
        dec2 = self.dec2(self.dec2_se(torch.cat([dec3, conv2], 1)))  #1

        x_no_empty = self.logits_no_empty(dec2)
        dec0_add_center = torch.cat((
            dec2,
            F.upsample(center_64, scale_factor=128, mode='bilinear')), 1)
        x_final = self.logits_final(dec0_add_center)

        return center_fc, x_no_empty, x_final