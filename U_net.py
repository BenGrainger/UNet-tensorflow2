import matplotlib.pyplot as plt
import os
import random
import math
import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))
print('number of GPUs avail:', len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate

from tensorflow.keras.models import Model

from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.metrics import RootMeanSquaredError

from tensorflow.keras.optimizers import Adam, SGD, RMSprop


#import sys
#sys.path.append(r'U:\users\beng\repos\Malis-Loss-master\malis')
#from malis_keras import malis_loss3D

# Define custom loss and dice coefficient
def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)

    return (2.*intersection)/(K.sum(y_true_f*y_true_f)+K.sum(y_pred_f*y_pred_f)+eps)

def weighted_binary_crossentropy(zero_weight, one_weight):
    def _weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = K.binary_crossentropy(y_true, y_pred)

        weight_vector = y_true*one_weight+(1.-y_true)*zero_weight
        weighted_binary_crossentropy = weight_vector*binary_crossentropy

        return K.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy
    
class SampleImageCallback(Callback):

    def __init__(self, model, sample_data, model_path, save=False):
        self.model = model
        self.sample_data = sample_data
        self.model_path = model_path
        self.save = save

    def on_epoch_end(self, epoch, logs={}):
        sample_predict = self.model.predict_on_batch(self.sample_data)

        f=plt.figure(figsize=(16,8))
        plt.subplot(1,2,1)
        plt.imshow(self.sample_data[0,:,:,0,0], interpolation='nearest', cmap='gray')
        plt.title('Sample source')
        plt.axis('off');

        plt.subplot(1,2,2)
        plt.imshow(sample_predict[0,:,:,0,0], interpolation='nearest', cmap='magma')
        plt.title('Predicted target')
        plt.axis('off');

        plt.show()

        if self.save:
            plt.savefig(self.model_path + '/epoch_' + str(epoch+1) + '.png')

class Unet3D:

    def __init__(self,
                 shape=(64,64,8,1)):
        if isinstance(shape, str):
            shape = eval(shape)

        self.shape = shape # this is the shape of the image data

        input_tensor = Input(self.shape, name='input')

        self.model = self.unet_3D(input_tensor)

    def down_block_3D(self, input_tensor, filters):
        x = Conv3D(filters=filters, kernel_size=(3,3,3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv3D(filters=filters*2, kernel_size=(3,3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def up_block_3D(self, input_tensor, concat_layer, filters):
        x = Conv3DTranspose(filters, kernel_size=(2,2,2), strides=(2,2,2))(input_tensor)

        x = Concatenate()([x, concat_layer])

        x = Conv3D(filters=filters, kernel_size=(3,3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv3D(filters=filters*2, kernel_size=(3,3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def unet_3D(self, input_tensor, filters=32):
        d1 = self.down_block_3D(input_tensor, filters=filters)
        p1 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), data_format='channels_last')(d1)
        d2 = self.down_block_3D(p1, filters=filters*2)
        p2 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), data_format='channels_last')(d2)
        d3 = self.down_block_3D(p2, filters=filters*4)
        p3 = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), data_format='channels_last')(d3)

        d4 = self.down_block_3D(p3, filters=filters*8)

        u1 = self.up_block_3D(d4, d3, filters=filters*4)
        u2 = self.up_block_3D(u1, d2, filters=filters*2)
        u3 = self.up_block_3D(u2, d1, filters=filters)

        output_tensor = Conv3D(filters=1, kernel_size=(1,1,1), activation='sigmoid')(u3)

        return Model(inputs=[input_tensor], outputs=[output_tensor])

    def summary(self):
        return self.model.summary()

    def train(self,
              epochs,
              batch_size,
              train_generator,
              val_generator,
              model_path,
              model_name,
              optimizer='adam',
              learning_rate=0.001,
              loss='weighted_binary_crossentropy',
              metrics='dice',
              ckpt_period=1,
              save_best_ckpt_only=True,
              ckpt_path=None):

        class_weight_zero, class_weight_one = train_generator.class_weights()

        if loss == 'weighted_binary_crossentropy':
            loss = weighted_binary_crossentropy(class_weight_zero, class_weight_one)

        #if loss == 'malis':
        #    loss = malis_loss3d # batch of 1! only works on linux

        if metrics == 'dice':
            metrics = dice_coefficient

        if optimizer == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=[metrics])
                  
        if ckpt_path is not None:
            self.model.load_weights(ckpt_path)

        full_model_path = os.path.join(model_path, model_name)

        if not os.path.exists(full_model_path):
            os.makedirs(full_model_path)

        log_dir = full_model_path + '/Quality Control'
        ckpt_dir =  full_model_path + '/ckpt'

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        if save_best_ckpt_only:
            ckpt_name = ckpt_dir + '/' + model_name + '.hdf5'
        else:
            ckpt_name = ckpt_dir + '/' + model_name + '_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.hdf5'

        model_ckpt = ModelCheckpoint(ckpt_name,
                                     verbose=1,
                                     save_freq=ckpt_period,
                                     save_best_only=save_best_ckpt_only,
                                     save_weights_only=True)
                  
        sample_batch, __ = val_generator.__getitem__(random.randint(0, len(val_generator)))
        sample_img = SampleImageCallback(self.model, 
                                         sample_batch, 
                                         model_path)

        history = self.model.fit(train_generator,
                       validation_data=val_generator,
                       validation_steps=math.floor(len(val_generator)/batch_size),
                       epochs=epochs,
                       callbacks=[model_ckpt,
                                  sample_img])

        last_ckpt_name = ckpt_dir + '/' + model_name + '_last.hdf5'
        self.model.save_weights(last_ckpt_name)

        return history

    def _min_max_scaling(self, data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)

        return n/d













