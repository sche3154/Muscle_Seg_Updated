
from utils.dmri_io import load_data
from utils.utils import tms_sample_list
import os
import numpy as np

pred_root = '/home/sheng/Muscle_Seg_updated/results'
src_root = '/home/sheng/datasets/muscle/generated_results/all_data'
exp = 'TmsCoarseL1_UNet3D_50' ## might need to change

sample_list= tms_sample_list(isTrain=False)

dice_list = []

for sample in sample_list:
    # print(sample)

    target = load_data(os.path.join(pred_root, exp, sample+'_pred.nii.gz'))
    gt = load_data(os.path.join(src_root, sample+'_mask.nii.gz'))
    dice = 2*np.sum(gt*target)/np.sum(gt+target)

    dice_list.append(dice)

print(dice_list)
print(np.mean(dice_list), np.std(dice_list))



