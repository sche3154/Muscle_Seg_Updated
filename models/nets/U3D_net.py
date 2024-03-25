import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.nets.blocks import *

class U3DNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=6, batch_norm=True, cnum=32):
        super(U3DNet, self).__init__()
        print("UNet3D is created")
        self.in_channels = in_channels
        self.out_channels = out_channels
        activation = nn.ReLU(inplace=True)
        # activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling In_dim
        self.enc1_1 = Conv3D(in_channels, cnum, 3, 1, padding= 1, batch_norm=batch_norm, activation=activation)
        self.enc1_2 = Conv3D(cnum, cnum, 3, 2, padding= 1, batch_norm=batch_norm, activation=activation)
        # In_dim/2
        self.enc2_1 = Conv3D(cnum, 2 * cnum, 3, 1, padding= 1, batch_norm=batch_norm, activation=activation)
        self.enc2_2 = Conv3D(2 * cnum, 2 * cnum, 3, 2, padding=1, batch_norm=batch_norm, activation=activation)
        # In_dim/4
        self.enc3_1 = Conv3D(2 * cnum, 4 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.enc3_2 = Conv3D(4 * cnum, 4 * cnum, 3, 2, padding=1, batch_norm=batch_norm, activation=activation)
        # In_dim/8
        self.enc4_1 = Conv3D(4 * cnum, 8 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        self.enc4_2 = Conv3D(8 * cnum, 8 * cnum, 3, 2, padding=1, batch_norm=batch_norm, activation=activation)
    
        # Bridge In_dim/16
        self.bridge = Conv3D(8 * cnum, 16 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)

        # Up sampling In_dim/16
        self.dec1_1 = DeConv3D(2, 16 * cnum, 8 * cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec1_2 = Conv3D(16 * cnum, 8 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        # Up Sampling In_dim/8
        self.dec2_1 = DeConv3D(2, 8 * cnum, 4 * cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec2_2 = Conv3D(8 * cnum, 4 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        # Up Sampling In_dim/4
        self.dec3_1 = DeConv3D(2, 4 * cnum, 2 * cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec3_2 = Conv3D(4 * cnum, 2 * cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
        # Up Sampling In_dim/2
        self.dec4_1 = DeConv3D(2, 2 * cnum, cnum, 3, 2, padding=1, output_padding = 1, batch_norm=batch_norm, activation=activation)
        self.dec4_2 = Conv3D(2 * cnum, cnum, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)

        # Output In_dim
        self.out = nn.Sequential(Conv3D(cnum, out_channels, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
                                 ,Conv3D(out_channels, out_channels, 1, 1, padding=0, batch_norm=batch_norm, activation=None))
        
        # self.out = Conv3D(cnum, out_channels, 3, 1, padding=1, batch_norm=batch_norm, activation=activation)
                                 

    def forward(self, x, encoder_only=False):
        feat = []

        # x: b c w h d
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

        if encoder_only:
            return feat

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
