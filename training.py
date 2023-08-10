import imgaug.augmenters as iaa
import random
import matplotlib.pyplot as plt
import os
from data_generator import MultiPageTiffGenerator
from U_net import Unet3D
import numpy as np

training_source = r'S:\users\beng\automatic_segmentation\mitochondria_training_data\training\input'
training_target = r'S:\users\beng\automatic_segmentation\mitochondria_training_data\training\output'
batch_size = 1
patch_size = (128,128,8) 
training_shape = patch_size + (1,)
apply_data_augmentation = True
(gaussian_frequency, gaussian_sigma) = (0.5, 0.7)
(contrast_frequency, contrast_min, contrast_max) = (0.5, 0.4, 1.6)
(noise_frequency, scale_min, scale_max) = (0.5, 0, 0.05)
augmentations = [iaa.Sometimes(gaussian_frequency, iaa.GaussianBlur(sigma=(0, gaussian_sigma))),
                 iaa.Sometimes(contrast_frequency, iaa.LinearContrast((contrast_min, contrast_max))),
                 iaa.Sometimes(noise_frequency, iaa.AdditiveGaussianNoise(scale=(scale_min, scale_max)))]
add_elastic_deform = True
deform_params = (2,2,1)
validation_split_in_percent = 20
random_crop = True
downscaling_in_xy = 1
binary_target = False
floor = True
floor_factor = 0.9

train_generator = MultiPageTiffGenerator(training_source,
                                         training_target,
                                         batch_size=batch_size,
                                         shape=training_shape,
                                         augment=apply_data_augmentation,
                                         augmentations=augmentations,
                                         deform_augment=add_elastic_deform,
                                         deform_augmentation_params=deform_params,
                                         val_split=validation_split_in_percent/100,
                                         random_crop=random_crop,
                                         downscale=downscaling_in_xy,
                                         binary_target=binary_target,
                                         floor=floor,
                                         floor_factor=floor_factor,
                                         zxy=True)

val_generator = MultiPageTiffGenerator(training_source,
                                         training_target,
                                         batch_size=batch_size,
                                         shape=training_shape,
                                         val_split=validation_split_in_percent/100,
                                         random_crop=random_crop,
                                         downscale=downscaling_in_xy,
                                         binary_target=binary_target,
                                         floor=floor,
                                         floor_factor=floor_factor,
                                         zxy=True)

print('train and validation generators initialised')

model = Unet3D()
print('Unet intitialised')

epochs=1
batch_size=1
model_path=r'S:\users\beng\automatic_segmentation\models'
model_name='test_1'
full_model_path = os.path.join(model_path, model_name)
history = model.train(epochs=epochs,
            batch_size=batch_size,
            train_generator=train_generator,
            val_generator=val_generator,
            model_path=model_path,
            model_name=model_name)


np.save(os.path.join(r'S:\users\beng\automatic_segmentation\model_history', '{}_loss.npy'.format(model_name)), history.history['loss'])
np.save(os.path.join(r'S:\users\beng\automatic_segmentation\model_history', '{}_vall_loss.npy'.format(model_name)), history.history['val_loss'])