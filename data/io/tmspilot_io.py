from data.io.base_io import BaseIO
from utils.dmri_io import load_data, save_data
from data.io.tms_sample import TmsSample
import os
# /home/sheng/datasets/muscle/pilot
class TmsPilotIO(BaseIO):

    def __init__(self, opt):
        BaseIO.__init__(self, opt)

    def load_sample(self, index):

        thigh_path = os.path.join(self.root, index + '.nii.gz')

        thigh, thigh_affine = load_data(thigh_path, needs_affine =True)
        thigh_mask = None
        sample = TmsSample(index, thigh, thigh_mask, thigh_affine)

        sample.thigh_shape = thigh.shape

        return sample
    
    def save_sample(self, sample):

        value, affine, index, name = sample
        output_name = os.path.join(self.opt.results_dir, self.opt.name, '{}_{}.nii.gz'.format(index,name))
        save_data(value, affine=affine, output_name=output_name)

