from multi_affine.networks import *

def test_network_multi_affine():
    vol_size = [64,64,64]
    filters_enc = [64,128,128,128]
    num_fcl = 4
    n_hidden = 1000

    output = network_multi_affine(vol_size, filters_enc, num_fcl, n_hidden, True)

    #check shape input image
    assert output[0][0].shape[1] == vol_size[0]

    #check output shape conv layers
    for i in range(len(filters_enc)):
        assert output[0][i+1].shape[4] == filters_enc[i]
    #check output shape gap layer
    assert output[1].shape[1] == filters_enc[-1]
    #check output shape FCL
    for i in range(num_fcl):
        assert output[2][i+1].shape[1] == n_hidden

    #check output dimensions
    assert output[3].shape[1] == 12
    assert output[4].shape[1] == 64


