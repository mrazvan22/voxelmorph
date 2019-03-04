import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import keras.layers as KL
import sys
sys.path.append('../../pytools-lib')
sys.path.append('../../pynd-lib/')
from pynd import ndutils as nd
from ndutils import volcrop
sys.path.append('../../neuron')
sys.path.append('../../neuron-sandbox')
from neuron import models as nrn_models
from neuron import utils
from sandbox import plot as nrn_sandbox_plt
from sandbox import imputation as im


# example compile code for SparseVM :   
#model.compile(optimizer=Adam(lr=lr, epsilon= 5e-5), loss=[
#                                    Mask(model.get_layer("mask").output).sparse_conv_cc3D(atlas_mask), Mask(model.get_layer("mask").output).gradientLoss('l2'), ls.fake_loss()], loss_weights=[1.0, reg_param, 0])

class Mask(object):

    def __init__(self, mask):
        self.mask = mask

    def sparse_conv_cc3D(self, atlas_mask, conv_size = 15, sum_filter = 1, padding = 'same', activation = 'elu'):
        def loss(I, J):
            # pass in mask to class: e.g. Mask(model.get_layer("mask").output).sparse_conv_cc3D(atlas_mask),
            mask = self.mask
            # need the next two lines to specify channel for source image (otherwise won't compile)
            I = I[:,:,:,:,0]
            I = tf.expand_dims(I, -1)
            
            I2 = I*I
            J2 = J*J
            IJ = I*J
            input_shape = I.shape
            # want the size without the channel and batch dimensions
            ndims = len(input_shape) -2
            strides = [1] * ndims
            convL = getattr(KL, 'Conv%dD' % ndims)
            im_conv = convL(sum_filter, conv_size, padding=padding, strides=strides,kernel_initializer=keras.initializers.Ones())
            im_conv.trainable = False
            mask_conv = convL(1, conv_size, padding=padding, use_bias=False, strides=strides,kernel_initializer=keras.initializers.Ones())
            mask_conv.trainable = False

            combined_mask = mask*atlas_mask
            u_I, out_mask_I, not_used, conv_mask_I = im.conv_block(I, mask, im_conv, mask_conv, 'u_I')
            u_J, out_mask_J, not_used, conv_mask_J = im.conv_block(J, atlas_mask, im_conv, mask_conv, 'u_J')
            not_used, not_used_mask, I_sum, conv_mask = im.conv_block(I, combined_mask, im_conv, mask_conv, 'I_sum')
            not_used, not_used_mask, J_sum, conv_mask = im.conv_block(J, combined_mask, im_conv, mask_conv, 'J_sum')
            not_used, not_used_mask, I2_sum, conv_mask = im.conv_block(I2, combined_mask, im_conv, mask_conv, 'I2_sum')
            not_used, not_used_mask, J2_sum, conv_mask = im.conv_block(J2, combined_mask, im_conv, mask_conv, 'J2_sum')
            not_used, not_used_mask, IJ_sum, conv_mask = im.conv_block(IJ, combined_mask, im_conv, mask_conv, 'IJ_sum')
    
            cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*conv_mask
            I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*conv_mask
            J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*conv_mask
            cc = cross*cross / (I_var*J_var + 1e-2) 
            return -1.0 * tf.reduce_mean(cc)
        return loss

    def gradientLoss(self,penalty='l1'):
        def loss(y_true, y_pred):
            dy = tf.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :])
            dx = tf.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
            dz = tf.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])

            if (penalty == 'l2'):
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz
            d = tf.reduce_mean(dx)+tf.reduce_mean(dy)+tf.reduce_mean(dz)
            return d/3.0

        return loss

             
