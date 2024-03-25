from data.processing.base_processing import BaseProcessing
from utils.patch_operations import concat_matrices
import numpy as np

class TmsProcessing(BaseProcessing):

    def __init__(self, opt):

        BaseProcessing.__init__(self, opt)
        self.kernel_size = (128,128,16)
        self.stride_size = (64,64,12)

    def preprocessing(self, sample):

        thigh_data = sample.thigh_data.squeeze()
        thigh_data[thigh_data<=1500] = 0.
        coarse_mask = np.zeros(thigh_data.shape)
        coarse_mask[thigh_data>1500] = 1.

        thigh_mask = sample.thigh_mask.squeeze()
        thigh_mask[thigh_mask>0.5] = 1.
        thigh_mask[thigh_mask<=0.5] = 0.

        print('InstanceNorm2D Slices the {:}'.format(sample.index))
        thigh_data_norm = self.instance_norm_2DSlices(thigh_data, coarse_mask)

        print('Bounding the {:}'.format(sample.index))
        x_1_t, x_2_t, y_1_t, y_2_t, z_1_t, z_2_t =  self.find_bounding_box_3D(coarse_mask)
        # print(x_1_t, x_2_t, y_1_t, y_2_t, z_1_t, z_2_t)
        x_1, x_2, y_1, y_2, z_1, z_2 = self.find_bounding_box_3D(thigh_mask)
        # print(x_1, x_2, y_1, y_2, z_1, z_2)
        bb = [x_1_t, x_2_t, y_1_t, y_2_t, z_1_t, z_2_t]
        bb = [max(x_1-16, x_1_t), min(x_2+16,x_2_t), max(y_1-16, y_1_t), min(y_2+16, y_2_t), z_1, z_2]
        
        thigh_data_bounded = thigh_data_norm[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        thigh_mask_bounded = thigh_mask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
        coarse_mask_bounded = coarse_mask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]

        sample.bb = bb
        sample.bb_shape = thigh_data_bounded.shape
        sample.coarse_mask = coarse_mask_bounded
        # print(thigh_data_bounded.shape, thigh_mask_bounded.shape)

        if self.opt.input_patch_size > 0:

            print('Patching the {:}'.format(sample.index))
            # thigh_mask_patches, mask_coords = self.patch_generation(thigh_mask_bounded
            #                                                                    , self.kernel_size, self.stride_size, three_dim=True)
            # sample.thigh_mask_patches = thigh_mask_patches
            # sample.coord = mask_coords

            # thigh_data_patches = self.patch_generation_by_coords(thigh_data_norm, mask_coords)
            # sample.thigh_data_patches = thigh_data_patches

            thigh_data_patches, thigh_coords = self.patch_generation(thigh_data_bounded
                                                                               , self.kernel_size, self.stride_size, three_dim=True)
            sample.thigh_data_patches = thigh_data_patches
            sample.coord = thigh_coords

            thigh_mask_patches = self.patch_generation_by_coords(thigh_mask_bounded, thigh_coords)
            sample.thigh_mask_patches = thigh_mask_patches
            
            # print(thigh_data_patches.shape, thigh_mask_patches.shape)

        # Updating data
        sample.thigh_data = thigh_data_bounded
        sample.thigh_mask = thigh_mask_bounded


    def postprocessing(self, preprocessed_sample, output_dict):
        
        target =  preprocessed_sample
        
        resulted_dict = {}
        for key, value in output_dict.items():
            resulted_img = value
            if self.opt.input_patch_size > 0:
                print('Un-patching the {:} of {:}'.format(key, target.index))
                # print(value.shape)
                resulted_img = concat_matrices(patches=value,
                                            image_size = target.bb_shape,
                                            window= self.kernel_size,
                                            overlap= self.stride_size,
                                            three_dim= True,
                                            coords=target.coord)
                
                if key is 'pred':
                    resulted_img[resulted_img > 0.5] = 1.
                    resulted_img[resulted_img <= 0.5] = 0.

                print('Un-bounding the {:} of {:}'.format(key, target.index))
                tmp = np.zeros((target.thigh_shape))
                bb = target.bb
                tmp[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] = resulted_img
                resulted_img = tmp

            resulted_dict[key] = resulted_img 

        return resulted_dict
    
    def instance_norm_2DSlices(self, image, mask=None):
        # (W,H,D)
        mask = mask if mask is not None else np.ones(image.shape)

        image[mask!=1] = 0

        for slice in range(mask.shape[2]):
            means = np.mean(image[...,slice])
            stds = np.std(image[...,slice])  

            image[...,slice] = (image[...,slice] - means)/stds  

        image[mask!=1] = 0
        return image
    
    def minmax_norm_2DSlices(self, image, mask=None):
        mask = mask if mask is not None else np.ones(image.shape)

        image[mask!=1] = 0

        for slice in range(mask.shape[2]):
            max_val = np.max(image[...,slice])
            min_val = np.min(image[...,slice])

            image[...,slice] = (image[...,slice] - min_val) / (max_val-min_val)

        image[mask!=1] = 0
        return image



