import torch
import torch.nn as nn

class Conv2D(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0):
        super(Conv2D, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv2d(input)
        x = self.batch_norm2d(x)
        x = self.activation(x)

        return x

class DeConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3
            , stride = 2, padding=1, output_padding = 1, dilation=1):
        super(DeConv2D, self).__init__()

        self.deconv2d = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride= stride
                            , padding= padding, output_padding = output_padding, dilation =dilation),  
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True))

    def forward(self, input):
        x = self.deconv2d(input)
        return x
    

class Conv3D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Conv3D, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm3d = torch.nn.BatchNorm3d(out_channels)

    def forward(self, input):
        x = self.conv3d(input)

        if self.batch_norm:
            x = self.batch_norm3d(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class DeConv3D(torch.nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size = 3
            , stride = 2, padding=1, output_padding = 1, dilation=1
            , groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(DeConv3D, self).__init__()

        self.conv3d = nn.Sequential(
                        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride= stride
                            , padding= padding, output_padding = output_padding, dilation =dilation),  
                        nn.BatchNorm3d(out_channels),
                        activation)

    def forward(self, input):
        x = self.conv3d(input)
        return x