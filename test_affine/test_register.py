from multi_affine.register import *
from multi_affine.networks import *

def test_register_image():
    img = 'test_affine/test_data/test_data/00001_vol.nii.gz'
    A_t = [17,0,32]
    A_b = [47,0,32]
    
    model = network_multi_affine((64,64,64), [4,4,4,4],1, 2)
    model.load_weights('test_affine/out_test_train/0.h5')

    out, mov, warp = register_image(img, A_t, A_b, model)
    print(out)
    assert out[1] == '00001'
    assert int(out[2]) == 30
   

def test_calculate_landmark_error():
    A_t = [17,0,0]
    A_b = [47,0,0]
 
    T = np.zeros((1,12))
    X_t = [47,0,0]
    X_b = [17,0,0]

    landmark_error = calculate_landmark_error(T, A_t, A_b, X_t, X_b)
    assert landmark_error - 30 < 1e-5


def test_save_png():
    im1 = np.zeros((64,64,64))
    im2 = np.ones((64,64,64))
    
    im = [] 
    labels = []
    for i in range(50):
        if i%2 == 0:
            im.append(im1)
            labels.append(1)
        else:
            im.append(im2)
            labels.append(2)
    
    save_png(im, labels, 'test_affine/test_data/test_png', 32, 64, 64, 2)
    save_png(im, labels, 'test_affine/test_data/test_png', 32, 64, 64, 0)
    save_png(im, labels, 'test_affine/test_data/test_png', 32, 64, 64, 1)

    files = os.listdir('test_affine/test_data/test_png')
    assert len(files) == 3
