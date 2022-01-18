# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
from tensorflow.python import matvec


class Scaling_loss(object):
    """
    Class to calculate the scaling loss as described in: https://link.springer.com/chapter/10.1007/978-3-030-60334-2_21

    Args:
        y_pred: (1,12) output of network containing T'=I-T

    Returns:
        loss: scaling loss
    """

    def loss(self, _, y_pred):
        #y_pred: warp_global
        vol_shape = y_pred.shape

        if vol_shape[-1] == 6:
            y_pred = K.permute_dimensions(y_pred,[1,0])
            phi = K.concatenate([y_pred[0:2],y_pred[3:5]],axis=1)
        if vol_shape[-1] == 12:
            y_pred = K.permute_dimensions(y_pred,[1,0])
            phi = K.concatenate([y_pred[0:3],y_pred[4:7],y_pred[8:11]],axis=1)
    
        phi += np.identity(phi.shape[0])
            
        s,u,v = tf.linalg.svd(phi)
        return K.sum(K.square(K.log(s)))
    

class Landmark_loss():
    """
    Class to calculate the landmark loss, which calculates the mean euclidian distance between the position
    of the landmarks after affine registration and the ground truth coordinates.
    For more explantation see: https://link.springer.com/chapter/10.1007/978-3-030-60334-2_21
    
    Args:
        A_t: image coordinates of the top landmark (here: crown)
        A_b: image coordinates of the bottom landmark (here: rump)
        size: halve times the size of the image dimension (assume square images)

    Returns:
        loss: the mean euclidian distance.
    """
    def __init__(self,A_t=[],A_b=[],size=32.0):
        self.A_t = A_t
        self.A_b = A_b
        self.size = size
        
    def loss(self,y_true,y_pred):
        error = []
        y_pred = K.concatenate((K.reshape(y_pred,[3,4]),K.zeros([1,4])),axis=0)+np.identity(4)
        X_t = [y_true[0,0],y_true[0,1],y_true[0,2]]
        X_b = [y_true[0,3],y_true[0,4],y_true[0,5]]
        x_t = tf.constant([self.size,self.size,self.size]) + matvec(y_pred,tf.concat((self.A_t - tf.constant([self.size,self.size,self.size]),tf.constant([1,],dtype='float32')), axis=0))[:-1]
        x_b = tf.constant([self.size,self.size,self.size]) + matvec(y_pred,tf.concat((self.A_b - tf.constant([self.size,self.size,self.size]),tf.constant([1,],dtype = 'float32')),axis = 0))[:-1]
        error = (tf.linalg.norm(X_t - x_t) + tf.linalg.norm(X_b - x_b))/2
        
        return error



class masked_NCC():
    """
    Class to calculate the local (over window) masked normalized cross correlation. In case of multiple atlases it returns the mean NCC.
    For details see: https://link.springer.com/chapter/10.1007/978-3-030-60334-2_21

    Args:
        N: total number of avaliable atlases (5* number of pregnancies)
        M: number of atlases taken into account in the loss
        win: window for local cc
        mask: mask that defines region where NCC is calculated
        multi_atlas: array containing all N avaliable atlases

    Returns:
        loss: the (mean) locak masked normalized cross correlation
    """

    def __init__(self, M = 7,win = None, eps = 1e-5, mask = 1,multi_atlas = 0, mode = 'affine'):
        self.win = win
        self.eps = eps
        self.mask = mask
        self.M = M
        self.multi_atlas = multi_atlas
        self.mode = mode

    def masked_ncc(self, I, J):
        # function to handle idx of random atlas

        J= tf.multiply(self.mask, J)

        cc = self.calculate_cross_correlation(I, J)
        cc = tf.multiply(self.mask, cc)

        return tf.reduce_sum(cc)/tf.reduce_sum(self.mask)

    def multi_ncc(self, I, J):
        # function to handle multi-atlas NCC.

        error = []
        indi = I[0,0:self.M,0,0,0]
        multi_atlas = tf.convert_to_tensor(self.multi_atlas, dtype=tf.float32)
        for i in range(self.M):
            idx = tf.cast(indi[i], tf.int32)
            atlas = tf.expand_dims(multi_atlas[idx,:,:,:,:], axis=0)
            J = tf.multiply(self.mask,J)
             
            cc = self.calculate_cross_correlation(atlas, J)
            cc = tf.multiply(self.mask,cc)
            
            error.append(tf.reduce_sum(cc)/tf.reduce_sum(self.mask))
       
        error = tf.convert_to_tensor(error)
        
        return K.sum(error)/self.M

    def calculate_cross_correlation(self, I, J):
        # calculate NCC: taken from Voxelmorph
        # set window size
        ndims = len(I.get_shape().as_list())-2
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)
      
     
        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J
       
        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
            padding = 'SAME'
            # compute local sums via convolution
            I_sum = conv_fn(I, sum_filt, strides, padding)
            J_sum = conv_fn(J, sum_filt, strides, padding)
            I2_sum = conv_fn(I2, sum_filt, strides, padding)
            J2_sum = conv_fn(J2, sum_filt, strides, padding)
            IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size
            
        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)
        return cc

    def loss(self, I, J):

        if self.mode == 'nonrigid':

            return -self.masked_ncc(I,J)

        else:
            return -self.multi_ncc(I,J)

    
