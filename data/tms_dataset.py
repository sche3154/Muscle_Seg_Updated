import torch
import numpy as np
from data.base_dataset import BaseDataset
from utils.utils import *
from data.io import create_io
from data.processing import create_prcoessing

class TmsDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self,opt)

        if self.opt.isTrain:
            subs = ['3','4','5','6','7','9','12','14','15']
        else:
            subs = ['10','11','15']

        all_data = list_dir(os.path.join(self.root))
        for data in all_data:
            if data.startswith('.'):
                all_data.remove(data)
            if 'noisy' in data:
                all_data.remove(data)

        self.sample_list = []
        for sub in subs:
            prefix = sub + '_'
            for data in all_data:
                if prefix in data and 'mask' not in data:
                    self.sample_list.append(data.split('.')[0])
        
        # self.sample_list = [self.sample_list[0]]
        # print(self.sample_list)
        self.data_io = create_io('tms', opt)
        self.processing = create_prcoessing('tms', opt)

    def __len__(self):

        return len(self.sample_list)

    def __getitem__(self, index):

        sub = self.sample_list[index]
        sample = self.data_io.load_sample(sub)
        self.processing.preprocessing(sample)
        if self.opt.isTrain is False:
            self.preprocessed_sample = sample

        
        thigh_data = torch.from_numpy(sample.thigh_data_patches).unsqueeze(1)
        thigh_mask = torch.from_numpy(sample.thigh_mask_patches).unsqueeze(1)

        if self.opt.isTrain:
            for patch in range(thigh_mask.shape[0]):
                randnum = np.random.rand()
                if randnum >= 0.5:
                    thigh_data[patch,...] = thigh_data[patch,...] - randnum * 0.5
                else:
                    thigh_data[patch,...] = thigh_data[patch,...] + randnum * 0.5

        # if self.opt.isTrain:
        #     for patch in range(thigh_mask.shape[0]):
        #         if np.random.rand() >= 0.5:
        #             thigh_data[patch,...] = addnoise(thigh_data[patch,...], noise_factor=0.2)
        #         if np.random.rand() >= 0.6:
        #             thigh_data[patch,...] = 1 - thigh_data[patch,...]

        
        # print(thigh_data.shape, thigh_mask.shape)
        return {'thigh_data': thigh_data, 'thigh_mask': thigh_mask}
    

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
                

def addnoise(inputs, noise_factor = 0.1):
    noise = inputs + torch.rand_like(inputs) * noise_factor
    noise_img = torch.clamp(noise,0.,1.)
  
    return noise_img

