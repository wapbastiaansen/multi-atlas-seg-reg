from multi_nonrigid.register import *
from multi_nonrigid.networks import *

def test_register_image():
    img = 'test_nonrigid/test_data/00001_vol_1_moved_affine.nii.gz'
    age = np.load('test_nonrigid/test_data/00001_vol_1_annotation.npz')['GA']
    enc = [16,32,32,32]
    dec = [32,32,32,32,32,16,16]
    atlas = np.load('test_nonrigid/test_data/atlas.npz')['atlas_vol'][0,:,:,:,:][np.newaxis,...]

    model = network_nonrigid((64,64,64), enc, dec)

    model.load_weights('test_nonrigid/out_test_train/0.h5')

    out, mov, phi, phi_inv = register_image_nonrigid(img, atlas, age,  model)

    assert out[1] == '00001'
    assert int(out[2]) == 30
    assert phi_inv == []
    assert mov.shape == (1,64,64,64,1)
    assert phi.shape == (1,64,64,64,3)



