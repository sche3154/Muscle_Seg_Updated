from models.base_model import BaseModel
from models.nets import create_net
import torch
import torch.nn as nn

class Dice(nn.Module):

    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, pred, gt):

        intersect = (gt*pred).sum() + 1e-8
        union = (gt+pred).sum() + 1e-8
        dice_loss = 1 - 2*intersect/union

        return dice_loss


class TmsModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['tms']
        self.loss_names = ['bce', 'dice', 'l1']

        self.net_tms = create_net(opt)
        self.sigmoid = nn.Sigmoid()

        if self.isTrain:
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = Dice()
            self.l1 = nn.L1Loss()
            self.optimizer = torch.optim.Adam(self.net_tms.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        
        self.thigh_data = input['thigh_data'].to(self.device).type(dtype= torch.cuda.FloatTensor)

        if self.opt.isTrain:
            self.thigh_mask = input['thigh_mask'].to(self.device).type(dtype= torch.cuda.FloatTensor)
            # print(self.thigh_data.dtype)
            print(self.thigh_data.shape, self.thigh_mask.shape)
        

    def forward(self):
        
        self.pred = self.net_tms(self.thigh_data)

        if self.opt.isTrain is False:
            self.pred = self.sigmoid(self.pred)
            # self.pred[self.pred > 0.5] = 1.
            # self.pred[self.pred <= 0.5] = 0.

            output = {'pred': self.pred, 'thigh_data': self.thigh_data
                    }
            # for k, v in output.items():
            #     print(v.shape)
            return output
    

    def backward(self):
        self.loss_bce = 0
        # if self.current_epoch < 20:
        self.loss_bce = self.bce(self.pred, self.thigh_mask)

        self.loss_dice = 0
        counter = 0
        for patch in range(self.thigh_mask.shape[0]):
            if self.thigh_mask[patch,...].sum() > 20000:
                counter += 1
                self.loss_dice += self.dice(self.sigmoid(self.pred[patch,...]), self.thigh_mask[patch,...])
        self.loss_bce = self.loss_bce
        self.loss_dice = self.loss_dice/(counter+1e-8)

        self.loss_l1 = 0
        self.loss_l1 = self.l1(self.sigmoid(self.pred), self.thigh_mask)

        print('bce: ', self.loss_bce)
        print('dice: ', self.loss_dice)
        print('l1:', self.loss_l1)

        self.loss_total = self.loss_bce + self.loss_dice + self.loss_l1
        self.loss_total.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        # update optimizer of the inpainting network
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()