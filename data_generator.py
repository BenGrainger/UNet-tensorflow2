import elasticdeform
import imgaug.augmenters as iaa
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from glob import glob
import math
import random

import tensorflow
from tensorflow.keras.utils import Sequence


# Define MultiPageTiffGenerator class
class MultiPageTiffGenerator(Sequence):

    def __init__(self,
                 source_path,
                 target_path,
                 batch_size=1,
                 shape=(128,128,32,1),
                 augment=False,
                 augmentations=[],
                 deform_augment=False,
                 deform_augmentation_params=(5,3,4),
                 val_split=0.2,
                 is_val=False,
                 random_crop=True,
                 downscale=1,
                 binary_target=False,
                 floor=True,
                 floor_factor=0.2,
                 xyzwhole=False,
                 zxy=False):

        self.source_path = source_path
        self.target_path = target_path
        self.source_dir_list = glob(os.path.join(source_path, '*'))
        self.target_dir_list = glob(os.path.join(target_path, '*'))

        self.source_dir_list.sort()
        self.target_dir_list.sort()
                     
        self.shape = shape
        self.batch_size = batch_size
                     
        self.augment = augment
        self.val_split = val_split
        self.is_val = is_val
        self.random_crop = random_crop
        self.downscale = downscale
        self.binary_target = binary_target
        self.deform_augment = deform_augment
        self.on_epoch_end()
        self.floor = floor
        self.floor_factor = floor_factor
        self.xyzwhole = xyzwhole
        self.zxy = zxy

        if self.augment:
            # pass list of augmentation functions 
            self.seq = iaa.Sequential(augmentations, random_order=True) # apply augmenters in random order
        if self.deform_augment:
            self.deform_sigma, self.deform_points, self.deform_order = deform_augmentation_params

    def __len__(self):
        if self.augment:
            augment_factor = 4
        else:
            augment_factor = 1

        
        num_of_imgs = 0
        for tiff_path in self.source_dir_list:
            num_of_imgs += tifffile.imread(tiff_path).shape[0]

        if len(self.source_dir_list) > 1:
            print(self.source_dir_list, 'is a tiff stack')
            xy_shape = tifffile.imread(self.source_dir_list[0]).shape[1:]
        else:
            print(self.source_dir_list, 'is a tiff volume')
            data = tifffile.imread(self.source_dir_list[0])
            if self.xyzwhole:
                x, y, z, whole = data.shape # this can change
            if self.zxy:
                z, x, y = data.shape
            xy_shape = (x, y)
            num_of_imgs = z

        if self.is_val:
            if self.random_crop:
                crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                volume = xy_shape[0] * xy_shape[1] * self.val_split * num_of_imgs
                print('__len__ output (validation):', math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale)))
                return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))
            else:
                print('__len__ output (validation):', math.floor(self.val_split * num_of_imgs / self.batch_size))
                return math.floor(self.val_split * num_of_imgs / self.batch_size)
        else:
            if self.random_crop:
                crop_volume = self.shape[0] * self.shape[1] * self.shape[2]
                volume = xy_shape[0] * xy_shape[1] * (1 - self.val_split) * num_of_imgs
                print('__len__ output:', math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale)))
                return math.floor(augment_factor * volume / (crop_volume * self.batch_size * self.downscale))

            else:
                print('__len__ output:', math.floor(self.val_split * num_of_imgs / self.batch_size))
                return math.floor(augment_factor*(1 - self.val_split) * num_of_imgs/self.batch_size)
        

    def __getitem__(self, idx):
        # note self.shape is referring to the block size and shape that will be within the batch
        source_batch = np.empty((self.batch_size,
                                 self.shape[0],
                                 self.shape[1],
                                 self.shape[2],
                                 self.shape[3]))
        target_batch = np.empty((self.batch_size,
                                 self.shape[0],
                                 self.shape[1],
                                 self.shape[2],
                                 self.shape[3]))

        # begin generating batches
        for batch in range(self.batch_size):
            # Modulo operator ensures IndexError is avoided
            stack_start = self.batch_list[(idx+batch*self.shape[2])%len(self.batch_list)]
            # print('stack start #:', stack_start[0], 'file:', self.source_dir_list[stack_start[0]])

            self.source = tifffile.imread(self.source_dir_list[stack_start[0]])
            if self.binary_target:
                self.target = tifffile.imread(self.target_dir_list[stack_start[0]]).astype(bool)
            else:
                self.target = tifffile.imread(self.target_dir_list[stack_start[0]])

            src_list = []
            tgt_list = []
            if len(self.source_dir_list) > 1: 
                for i in range(stack_start[1], stack_start[1]+self.shape[2]):
                    src = self.source[i]
                    src = transform.downscale_local_mean(src, (self.downscale, self.downscale))
                    if not self.random_crop:
                        src = transform.resize(src, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                    src = self._min_max_scaling(src)
                    src_list.append(src)

                    tgt = self.target[i]
                    tgt = transform.downscale_local_mean(tgt, (self.downscale, self.downscale))
                    if not self.random_crop:
                        tgt = transform.resize(tgt, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                    if not self.binary_target:
                        tgt = self._min_max_scaling(tgt)
                    tgt_list.append(tgt)
            
            else: # if tiff volume
                self.source = tifffile.imread(self.source_dir_list[stack_start[0]])
                self.target = tifffile.imread(self.target_dir_list[stack_start[0]])

                if self.xyzwhole:
                    x, y, z, whole = self.source.shape 
                    xy_shape = (x, y)
                    num_of_images = z
                    self.source = np.reshape(self.source.T, (num_of_images, xy_shape[0], xy_shape[1], 1))
                    self.target = np.reshape(self.target.T, (num_of_images, xy_shape[0], xy_shape[1], 1))
                if self.zxy:
                    z, x, y = self.source.shape
                    xy_shape = (x, y)
                    num_of_images = z
                    self.source = np.expand_dims(self.source, 3)
                    self.target = np.expand_dims(self.target, 3)

                for i in range(num_of_images):
                    src = self.source[i]
                    src = transform.downscale_local_mean(np.reshape(src, (xy_shape[0], xy_shape[1])), (self.downscale, self.downscale))
                    if not self.random_crop:
                        src = transform.resize(src, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                    src = self._min_max_scaling(src)
                    src_list.append(src)

                    tgt = self.target[i]
                    tgt = transform.downscale_local_mean(np.reshape(tgt, (xy_shape[0], xy_shape[1])), (self.downscale, self.downscale))
                    if not self.random_crop:
                        tgt = transform.resize(tgt, (self.shape[0], self.shape[1]), mode='constant', preserve_range=True)
                    if not self.binary_target:
                        tgt = self._min_max_scaling(tgt)
                    tgt_list.append(tgt)


            # check and assign crop volume axis
            if self.random_crop:
                if src.shape[0] == self.shape[0]:
                    x_rand = 0
                if src.shape[1] == self.shape[1]:
                    y_rand = 0
                if src.shape[0] > self.shape[0]:
                    x_rand = np.random.randint(src.shape[0] - self.shape[0])
                if src.shape[1] > self.shape[1]:
                    y_rand = np.random.randint(src.shape[1] - self.shape[1])
                if src.shape[0] < self.shape[0] or src.shape[1] < self.shape[1]:
                    raise ValueError('Patch shape larger than (downscaled) source shape')

            for i in range(self.shape[2]):
                if self.random_crop:
                    src = src_list[i]
                    tgt = tgt_list[i]
                    src_crop = src[x_rand:self.shape[0]+x_rand, y_rand:self.shape[1]+y_rand]
                    tgt_crop = tgt[x_rand:self.shape[0]+x_rand, y_rand:self.shape[1]+y_rand]
                else:
                    src_crop = src_list[i]
                    tgt_crop = tgt_list[i]

                source_batch[batch,:,:,i,0] = src_crop
                target_batch[batch,:,:,i,0] = tgt_crop

        if self.floor:
            target_batch = np.where(target_batch > self.floor_factor, 1, target_batch) # cremi floor_factor =0.2

        if self.augment:
            # On-the-fly data augmentation
            source_batch, target_batch = self.augment_volume(source_batch, target_batch)

            # Data augmentation by reversing stack
            if np.random.random() > 0.5:
                source_batch, target_batch = source_batch[::-1], target_batch[::-1]

            # Data augmentation by elastic deformation
            if np.random.random() > 0.5 and self.deform_augment:
                source_batch, target_batch = self.deform_volume(source_batch, target_batch)

            if not self.binary_target:
                target_batch = self._min_max_scaling(target_batch)

            return self._min_max_scaling(source_batch), target_batch

        else:
            return source_batch, target_batch

    def on_epoch_end(self):
        # Validation split performed here
        self.batch_list = []
        
        # Create batch_list of all combinations of tifffile and stack position
        if len(self.source_dir_list) > 1:
            for i in range(len(self.source_dir_list)):
                num_of_pages = tifffile.imread(self.source_dir_list[i]).shape[0]
                if self.is_val:
                    start_page = num_of_pages-math.floor(self.val_split*num_of_pages)
                    for j in range(start_page, num_of_pages-self.shape[2]):
                      self.batch_list.append([i, j])
                else:
                    last_page = math.floor((1-self.val_split)*num_of_pages)
                    for j in range(last_page-self.shape[2]):
                        self.batch_list.append([i, j])
        else:
            volume = tifffile.imread(self.source_dir_list[0])
            num_of_pages = volume.shape[2]
            if self.is_val:
                start_page = num_of_pages - math.floor(self.val_split * num_of_pages)
                for j in range(start_page, num_of_pages - self.shape[2]):
                    self.batch_list.append([0, j])
            else:
                last_page = math.floor((1 - self.val_split) * num_of_pages)
                for j in range(last_page - self.shape[2]):
                    self.batch_list.append([0, j])



        if self.is_val and (len(self.batch_list) <= 0):
            raise ValueError('validation_split too small! Increase val_split or decrease z-depth')

        random.shuffle(self.batch_list)

    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)

        return n/d

    def class_weights(self):
        ones = 0
        pixels = 0

        if len(self.target_dir_list) > 1:
            for i in range(len(self.target_dir_list)):
                tgt = tifffile.imread(self.target_dir_list[i])
                ones += np.sum(tgt)
                pixels += tgt.shape[0]*tgt.shape[1]*tgt.shape[2]
        else:
            if self.zxy:
                tgt = tifffile.imread(self.target_dir_list[0])
                for i in range(tgt.shape[0]):
                    ones += np.sum(tgt)
                    pixels += tgt.shape[0]*tgt.shape[1]
            # add if self.xyzwhole
            
        p_ones = ones/pixels
        p_zeros = 1-p_ones

        # Return swapped probability to increase weight of unlikely class
        return p_ones, p_zeros


    def deform_volume(self, src_vol, tgt_vol):
        [src_dfrm, tgt_dfrm] = elasticdeform.deform_random_grid([src_vol, tgt_vol],
                                                                axis=(1, 2, 3),
                                                                sigma=self.deform_sigma,
                                                                points=self.deform_points,
                                                                order=self.deform_order)
        if self.binary_target:
            tgt_dfrm = tgt_dfrm > 0.1

        return self._min_max_scaling(src_dfrm), tgt_dfrm

    def augment_volume(self, src_vol, tgt_vol):

        src_vol_aug = np.empty(src_vol.shape)
        tgt_vol_aug = np.empty(tgt_vol.shape)

        for i in range(src_vol.shape[3]):
            src_aug_z, tgt_aug_z = self.seq(images=src_vol[:,:,:,i,0].astype('float16'),
                                                                      segmentation_maps=np.expand_dims(tgt_vol[:,:,:,i,0].astype('uint8'), axis=-1))
            src_vol_aug[:,:,:,i,0] = src_aug_z
            tgt_vol_aug[:,:,:,i,0] = np.squeeze(tgt_aug_z)
        return self._min_max_scaling(src_vol_aug), tgt_vol_aug

    def sample_augmentation(self, idx):
        # observing the augmentation 
        src, tgt = self.__getitem__(idx)

        src_aug, tgt_aug = self.augment_volume(src, tgt)

        if self.deform_augment:
            src_aug, tgt_aug = self.deform_volume(src_aug, tgt_aug) 
            #src_aug, tgt_aug = self.deform_volume(src, tgt)

        return src_aug, tgt_aug



























