from multi_affine.eval import *
import numpy as np
import nibabel as nib



def test_apply_affine_inv():
    T = np.zeros((1,12))
    img = nib.load('test_affine/test_data/test_data/00001_vol.nii.gz').get_fdata()

    seg = apply_affine_inv(T, img, True)

    assert np.sum(img - seg) < 1e-5

def test_calculate_dice():
    X = np.ones((2,2))
    Y = np.ones((2,2))
    Y[0,0] = 0
    
    dice = calculate_dice(X,X)
    assert np.sum(dice - 1) < 1e-5

    dice = calculate_dice(X,Y)
    assert np.sum(dice - 6/7) < 1e-5

    dice = calculate_dice(Y,X)
    assert np.sum(dice - 6/7) < 1e-5

def test_calculate_ev_error():
    Seg_affine_inv = nib.load('test_affine/test_data/seg_dir/seg_00001_1.nii.gz').get_fdata()

    EV_VR = 32.77
    vox_size = 1
    factor = 4

    error,EV_AI = calculate_ev_error(Seg_affine_inv, EV_VR, vox_size, factor)

    assert error < 1e-4
    assert EV_AI - 32.77 < 1e-5

def test_majority_voting():
    seg = np.zeros((3,2,2))
    seg[0, 0, 0] = 1
    seg[0, 0, 1] = 1
    seg[1, 0, 0] = 1
    seg[1, 1, 0] = 1
    seg[2, 1, 0] = 1
    seg[2, 1, 1] =1

    major_seg = majority_voting(seg)

    assert major_seg[0,0] == 1.0
    assert np.sum(major_seg) == 2.0

    seg = np.zeros((2,2,2))
    seg[0, 0, 0] = 1
    seg[0, 1, 0] = 1
    seg[1, 0, 0] = 1
    seg[1, 0, 1] = 1

    major_seg=majority_voting(seg)

    assert major_seg[0,0] == 1.0
    assert np.sum(major_seg) == 1.0



