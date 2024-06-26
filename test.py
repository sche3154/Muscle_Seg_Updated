import torch
import numpy as np
import random
import time
import os 

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.utils import mkdirs

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    torch.cuda.empty_cache()
    opt = TestOptions().parse()  # get test options
    results_dir = os.path.join(opt.results_dir, opt.name)
    mkdirs(results_dir)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    for i, data in enumerate(dataset):

        start_time = time.time()

        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        if opt.input_patch_size > 0: # now can handle patches
            pred_list = []
            patch_nums = len(data[next(iter(data))].squeeze(0))
            # print(patch_nums)
            outputs = []

            for j in range(0, patch_nums, opt.input_patch_size):
                data_patched = {}
                for key, value in data.items():
                    value = value.squeeze(0)
                    data_patched[key] = value[j:min(j+opt.input_patch_size, patch_nums),...]
                    # print(data_patched[key].shape)
                model.set_input(data_patched)
                output = model.test()  # run inference
                outputs.append(output)

            dataset.dataset.postprocessing(outputs, counter= i+1)

        else:
            pass


    print('End inference')