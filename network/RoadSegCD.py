"""
Codes of Dblock based on https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial

import math

nonlinearity = partial(F.relu,inplace=True)

def resize(tensor, newsize):
    return F.interpolate(
        tensor, size=newsize, mode='bilinear', align_corners=True)

class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
class RoadSegCD(nn.Module):
    def __init__(self, task1_classes=1, task2_classes=1):
        super(RoadSegCD, self).__init__()

        filters = [64, 128, 256]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        
        self.dblock = Dblock(256)
        self.up = self.conv_stage(512,256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, task1_classes, 3, padding=1)

        self.a_decoder3 = DecoderBlock(filters[2], filters[1])
        self.a_decoder2 = DecoderBlock(filters[1], filters[0])
        self.a_decoder1 = DecoderBlock(filters[0], filters[0])

        self.a_finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.a_finalrelu1 = nonlinearity
        self.a_finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.a_finalrelu2 = nonlinearity
        self.a_finalconv3 = nn.Conv2d(32, task2_classes, 3, padding=1)

        for m in [
            self.a_finaldeconv1,
            self.a_finalconv2,
        ]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
        if useBN:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              #nn.LeakyReLU(0.1),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              #nn.LeakyReLU(0.1),
              nn.ReLU(),
            )
        else:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU()
            )

    def forward(self, x_image, x_angles):
        # Encoder
        x_image = self.firstconv(x_image)
        x_image = self.firstbn(x_image)
        x_image = self.firstrelu(x_image)
        x_image = self.firstmaxpool(x_image)

        x_angles = self.firstconv(x_angles)
        x_angles = self.firstbn(x_angles)
        x_angles = self.firstrelu(x_angles)
        x_angles = self.firstmaxpool(x_angles)

        e1 = self.encoder1(x_image)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        a_e1 = self.encoder1(x_angles)
        a_e2 = self.encoder2(a_e1)
        a_e3 = self.encoder3(a_e2)
        

        e3_merge = self.up(torch.cat((e3, a_e3), dim=1))

        # Center
        e3 = self.dblock(e3)
        e3_merge = self.dblock(e3_merge)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        a_d3 = self.a_decoder3(e3_merge) + e2 + a_e2
        a_d2 = self.a_decoder2(a_d3) + e1 + a_e1
        a_d1 = self.a_decoder1(a_d2)

        a_out = self.a_finaldeconv1(a_d1)
        a_out = self.a_finalrelu1(a_out)
        a_out = self.a_finalconv2(a_out)
        a_out = self.a_finalrelu2(a_out)
        a_out = self.a_finalconv3(a_out)

        # out: outputs a_out: pred_vecmaps, i.e. angles

        return F.sigmoid(out), F.sigmoid(a_out)

