import torch.nn as nn
from models.nets.U3D_net import U3DNet

class TmsNet(nn.Module):

    def __init__(self, opt):
        super(TmsNet, self).__init__()
        self.opt = opt
        self.net = U3DNet(opt.input_nc, opt.output_nc)


    def forward(self, x):
        
        out = self.net(x)

        return out