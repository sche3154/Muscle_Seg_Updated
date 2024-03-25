import torch
import numpy as np
from data.base_dataset import BaseDataset
from utils.utils import *
from data.io import create_io
from data.processing import create_prcoessing

class TmsPilotDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self,opt)

        # self.sample_list = ['thigh_left_roi', 'thigh_right_roi']
        self.sample_list = ['thigh_left_roi_n4', 'thigh_right_roi_n4']
        # print(self.sample_list)
        self.data_io = create_io('tmspilot', opt)
        self.processing = create_prcoessing('tmspilot', opt)
        # self.processing = create_prcoessing('tmspilotV2', opt)

    def __len__(self):

        return len(self.sample_list)

    def __getitem__(self, index):

        sub = self.sample_list[index]
        sample = self.data_io.load_sample(sub)
        self.processing.preprocessing(sample)
        if self.opt.isTrain is False:
            self.preprocessed_sample = sample

        thigh_data = torch.from_numpy(sample.thigh_data_patches).unsqueeze(1)
      
        # print(thigh_data.shape, thigh_mask.shape)
        return {'thigh_data': thigh_data}
    

    def postprocessing(self, outputs, counter):
        #  outputs: list of small dicts of outputs
        out_dict = {key:[] for key in outputs[0].keys()}

        for output in outputs:
            for k, v in output.items():
                v = v.squeeze(1)
                # print(v.shape)
                v = v.detach().cpu().numpy()
                out_dict[k].append(v)

        for key in out_dict.keys(): 
            # Concatenaing different ouputs seperately
            out_dict[key] = np.concatenate(out_dict[key], axis=0) 
            # print(out_dict[key].shape)

        resulted_dict = self.processing.postprocessing(self.preprocessed_sample, out_dict)

        if counter % self.opt.save_prediction == 0:
            for key in resulted_dict.keys():
                name = key
                self.data_io.save_sample((resulted_dict[key], self.preprocessed_sample.affine
                                          , self.preprocessed_sample.index, name))