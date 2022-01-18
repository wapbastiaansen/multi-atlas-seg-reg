# third party
from keras.models import Model
import keras.layers as KL
from keras.layers import LeakyReLU, Input
from keras.initializers import he_uniform

# import neuron layers, which will be useful for Transforming.
import neuron.layers as nrn_layers


def network_multi_affine(vol_size,filters_enc,num_fcl,n_hidden,test=False,filter_size=3,src_feats=1,full_size=True, indexing='ij'):
    """
    Function that creates the architecture of the multi affine network.

    Args:
        vol_size: shape of an image
        filters_enc: list were each entry gives the number of filters of each layer of the encoder
        num_fcl: number of fully conntected layers in the architecture
        n_hidden: number of neurons in each fcl

    Returns:
        Multi-atlas affine network architecture
    """

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    # inputs
    src = Input(shape=[*vol_size, src_feats])
    x_in = src
    
    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(filters_enc)):
        x_enc.append(conv_block_affine(x_enc[-1], filters_enc[i], filter_size, 2))
       
    #gap to FCL
    x_g=global_average_pooling(x_enc[-1])
    
    x_fcl=[x_g]
    for i in range(num_fcl):
        x_fcl.append(FCL(x_fcl[-1],n_hidden))
    
    dim_out=ndims*ndims+ndims 
    #number of affine parameters
    
    warp_global=KL.Dense(dim_out,name='warp_global',kernel_initializer=he_uniform())(x_fcl[-1])
    y_global = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing,name='y_global')([src, warp_global])
    
    model = Model(inputs=[src], outputs=[y_global, warp_global])
    if test == False:
        return model
    else:
        return [x_enc, x_g, x_fcl, warp_global, y_global]


def FCL(x_in, n_hidden):
    """
    Implementation of a fully connected layer. The initlializer is he_uniform,
    activation is leakyrelu.

    Args:
        x_in: input vector
        n_hidden: number of neurons in the layer

    Returns:
        x_out: output vector
    """
    x_out = KL.Dense(n_hidden,kernel_initializer=he_uniform())(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    
    return x_out

def global_average_pooling(x_in):
    """
    Global average pooling layer
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    gap = getattr(KL, 'GlobalAveragePooling%dD' % ndims)
    x_out = gap()(x_in)

    return x_out

def conv_block_affine(x_in, nf, filter_size, strides=1):
    """
    Implementation of convolutional block, including (down-sampling) convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, filter_size, padding='same',
                 kernel_initializer=he_uniform(), strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

