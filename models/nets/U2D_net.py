import torch
import torch.nn as nn
from models.nets.blocks import *

class U2DNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, cnum=32):

        super(U2DNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        activation = nn.ReLU(inplace=True)

        # Down sampling In_dim
        self.enc1_1 = Conv2D(in_channels, cnum, 3, 1, padding= 1)
        self.enc1_2 = Conv2D(cnum, cnum, 3, 2, padding= 1)
        # In_dim/2
        self.enc2_1 = Conv2D(cnum, 2 * cnum, 3, 1, padding= 1)
        self.enc2_2 = Conv2D(2 * cnum, 2 * cnum, 3, 2, padding=1)
        # In_dim/4
        self.enc3_1 = Conv2D(2 * cnum, 4 * cnum, 3, 1, padding=1)
        self.enc3_2 = Conv2D(4 * cnum, 4 * cnum, 3, 2, padding=1)
        # In_dim/8
        self.enc4_1 = Conv2D(4 * cnum, 8 * cnum, 3, 1, padding=1)
        self.enc4_2 = Conv2D(8 * cnum, 8 * cnum, 3, 2, padding=1)

        # Bridge In_dim/16
        self.bridge = Conv2D(8 * cnum, 16 * cnum, 3, 1, padding=1)

        # Up sampling In_dim/16
        self.dec1_1 = DeConv2D(2, 16 * cnum, 8 * cnum, 3, 2, padding=1, output_padding = 1)
        self.dec1_2 = Conv2D(16 * cnum, 8 * cnum, 3, 1, padding=1)
        # Up Sampling In_dim/8
        self.dec2_1 = DeConv2D(2, 8 * cnum, 4 * cnum, 3, 2, padding=1, output_padding = 1)
        self.dec2_2 = Conv2D(8 * cnum, 4 * cnum, 3, 1, padding=1)
        # Up Sampling In_dim/4
        self.dec3_1 = DeConv2D(2, 4 * cnum, 2 * cnum, 3, 2, padding=1, output_padding = 1)
        self.dec3_2 = Conv2D(4 * cnum, 2 * cnum, 3, 1, padding=1)
        # Up Sampling In_dim/2
        self.dec4_1 = DeConv2D(2, 2 * cnum, cnum, 3, 2, padding=1, output_padding = 1)
        self.dec4_2 = Conv2D(2 * cnum, cnum, 3, 1, padding=1)

        # Output In_dim
        self.out = Conv2D(cnum, out_channels, 3, 1, padding=1)

    def forward(self, x):

        # x: b c w h
        # Down sampling
        down_1 = self.enc1_1(x)
        pool_1 = self.enc1_2(down_1)

        down_2 = self.enc2_1(pool_1)
        pool_2 = self.enc2_2(down_2)
            
        down_3 = self.enc3_1(pool_2)
        pool_3 = self.enc3_2(down_3)
            
        down_4 = self.enc4_1(pool_3)
        pool_4 = self.enc4_2(down_4)
            
        # print('pool shape', pool_1.shape, pool_2.shape, pool_3.shape, pool_4.shape)

        # Bridge
        bridge = self.bridge(pool_4)

        # Up sampling
        trans_1 = self.dec1_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.dec1_2(concat_1)

        trans_2 = self.dec2_1(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.dec2_2(concat_2)

        trans_3 = self.dec3_1(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.dec3_2(concat_3)

        trans_4 = self.dec4_1(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.dec4_2(concat_4)
 
        out = self.out(up_4)

        return out