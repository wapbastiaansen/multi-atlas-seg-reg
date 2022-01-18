from multi_nonrigid.networks import *

def test_network_nonrigid():
    vol_size = [64,64,64]
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    diffeomorphic = False

    output = network_nonrigid(vol_size, enc_nf, dec_nf, diffeomorphic=diffeomorphic, test=True)

    #check shape output image
    assert output[0][0].shape[1] == vol_size[0]

    #check output shape conv layers
    for i in range(len(enc_nf)):
        assert output[0][i+1].shape[4] == enc_nf[i]

    #check shape output dec
    assert output[1].shape[1] == vol_size[0]

    #check output shape warp
    assert output[2].shape[4] == len(vol_size)
    assert output[2].shape[1] == 64

    #check output shape img
    assert output[3].shape[1] == 64

    output = network_nonrigid(vol_size, enc_nf, dec_nf, diffeomorphic=True, test=True)

    assert output[2].shape[4] == len(vol_size)
    assert output[2].shape[1] == 64

    assert output[3].shape[4] == len(vol_size)
    assert output[3].shape[1] == 64

    assert output[4].shape[4] == output[3].shape[4]
    assert output[4].shape[1] == output[3].shape[1]

    assert output[5].shape[1] == 64
    
