# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
    
        
class Masked_MSE():
    """
    Class to calculate the masked MSE loss as defined in https://arxiv.org/abs/2005.06368

    Args:
        mask: binary mask to indicate where the loss should be calculate. Here this is used to mask the varying background in the target image.

    Returns:
        masked MSE loss

    #TODO: test
    """
    def __init__(self,mask=1):
        self.mask=mask
        
    def loss(self, y_true, y_pred):
        squared=K.square(y_pred-y_true)
        masked=self.mask*squared
        
        return K.sum(masked)/K.sum(self.mask)
    
class zeros_loss():
    def loss(self, y_true, y_pred):
        return tf.zeros([1,],dtype=tf.float32)

class grad_3D():
    """
    N-D gradient loss taken from voxelmorph as diffusion regularization
    (implementation is the orginal one and therefor not tested)
    
    Args:
        mask_sum: the number of voxels over which we calculate the field. We normalize the penalty over this
        See: https://link.springer.com/chapter/10.1007/978-3-030-60334-2_21

    """

    def __init__(self, mask_sum=1, penalty='l1'):
        self.mask_sum=mask_sum
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1] 
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_sum(f * f)/self.mask_sum for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)


    
class Masked_NMI():
    """
    Normalized mutual information of two images within given mask
    derived from voxelmorph
    """

    #tODO: test
    def __init__(self, mask=None, num_bins=20, max_clip=1):
        self.mask=mask
        self.num_bins=num_bins
        self.max_clip=max_clip
        
    def NMI(self, y_true, y_pred):
        """ soft mutual info """
        bin_centers=np.linspace(0,1,self.num_bins+1)
        bin_centers=(bin_centers[1:]-bin_centers[:-1])/2+bin_centers[1:]
        vol_bin_centers = K.variable(bin_centers)
        #num_bins = len(bin_centers)
        sigma = np.mean(np.diff(bin_centers))

        preterm = K.variable(1 / (2 * np.square(sigma)))
        y_pred = K.clip(y_pred, 0, self.max_clip)
        y_true = K.clip(y_true, 0, self.max_clip)
        
        # mask images and remove zero pixels (ie outside mask)
        #mask=self.mask>0
        #y_true=tf.multiply(y_true,self.mask)
        #y_pred=tf.multiply(y_pred,self.mask)
        y_true=tf.dynamic_partition(y_true,self.mask,2)
        y_true=y_true[1]
        y_pred=tf.dynamic_partition(y_pred,self.mask,2)
        y_pred=y_pred[1]
        
        # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
        #y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
        y_true=K.expand_dims(y_true,0)
        y_true = K.expand_dims(y_true, 2)
        #y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
        y_pred=K.expand_dims(y_pred,0)
        y_pred = K.expand_dims(y_pred, 2)
        print(y_true.shape)
        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(vol_bin_centers, o)
        
        # compute image terms
        I_a = K.exp(- preterm * K.square(y_true  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- preterm * K.square(y_pred  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)
        
        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1)

        return mi

    def loss(self, y_true, y_pred):
        return -self.NMI(y_true, y_pred)
