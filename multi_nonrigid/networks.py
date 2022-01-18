from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Lambda, Cropping3D, Cropping2D, ZeroPadding2D, ZeroPadding3D
from keras.initializers import RandomNormal
import tensorflow as tf
import neuron.layers as nrn_layers


def network_nonrigid(vol_size, enc_nf, dec_nf, src_feats=1, tgt_feats=1, diffeomorphic=False, int_steps=7, indexing='ij', test=False):
    """
    Function that creates the architecture of the nonrigid network. 

    Args:
        vol_size: shape of the image
        enc_nf: list where each entry gives the number of filters of each layer of the encoder
        dec_nf: list where each entry gives the number of filters of each layer of the decoder. Not if len(dec_nf)=7 2 conv layers at full resolution are applied
        diffeomorphic: is set to true diffeomorphic model is used.
        int_steps: number of integration steps for the diffeomorphic model.

    Returns:
        Multi-atlas nonrigid network achitecture with output:
        [y, warp_local, inv_warp_local]
        y: moved image
        warp_local: deformation field that maps I to A
        inv_warp_local: deformation field that maps A to I
    """
    
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    
    # inputs
    src = Input(shape=[*vol_size, src_feats])
    tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))
    
    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[0]])

    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[5])
        x = conv_block(x, dec_nf[6])
    
    Conv = getattr(KL, 'Conv%dD' % ndims)

    if diffeomorphic == True:
        velocity_field = Conv(ndims, kernel_size=3, padding='same', name='velocity_field', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

        warp_local = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(velocity_field)

        
        rev_velocity_field = Negate()(velocity_field)
        inv_warp_local = nrn_layers.VecInt(method='ss', name='neg_flow-int', int_steps=int_steps)(rev_velocity_field)

        y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='y')([src, warp_local])

        model = Model(inputs=[src, tgt], outputs=[y, warp_local, inv_warp_local])
    else:
        # transform the results into a flow field.
        warp_local = Conv(ndims, kernel_size=3, padding='same', name='warp_local',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='y')([src, warp_local])
        # prepare model
        model = Model(inputs=[src, tgt], outputs=[y, warp_local])
    
    if test == False:
        return model
    else:
        if diffeomorphic == True:
            return [x_enc, x, velocity_field, warp_local, inv_warp_local, y]
        else:
            return [x_enc, x, warp_local, y]


def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

class Negate(Layer):
    """ 
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape


