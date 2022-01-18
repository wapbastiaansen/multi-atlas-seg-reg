from multi_affine.datagenerators import *
import numpy as np
from pytest import raises
import matplotlib.pyplot as plt
import glob
import os

def test_indicator():
 
    age_atlas = np.array([56, 56, 59, 63, 65, 66, 70, 71, 74, 76, 77, 80, 84, 86, 87]).reshape((15,1))
    atlas_list = ['test/atlas_00001.npz','test/atlas_00003.npz','test/atlas_00002.npz', 'test/atlas_00001.npz','test/atlas_00003.npz','test/atlas_00002.npz','test/atlas_00001.npz','test/atlas_00002.npz','test/atlas_00003.npz','test/atlas_00003.npz','test/atlas_00001.npz','test/atlas_00002.npz','test/atlas_00001.npz','test/atlas_00003.npz','test/atlas_00002.npz']

    ini = indicator(64,1,age_atlas, atlas_list)
    assert ini[3] == 1 and np.sum(ini) == 1

    ini = indicator(56,1,age_atlas, atlas_list)
    assert ini[0] == 1 and np.sum(ini) == 1

    ini = indicator(64,2,age_atlas, atlas_list)
    assert ini[3] == 1 and ini[4] == 1 and np.sum(ini) == 2

    ini = indicator(75,3,age_atlas,atlas_list)
    assert ini[7] == 1 and ini [10] == 1 and ini[8] ==1 and np.sum(ini) == 3
    
    with raises(AssertionError):
        indicator(30,47,age_atlas, atlas_list)
    
def test_load_volfile():
    
    X = load_volfile('test_affine/test_data/atlas.npz',np_var='atlas_vol')
    assert np.sum(X) == 0

    X = load_volfile('test_affine/test_data/test_data/00001_vol.nii.gz')
    assert np.sum(X) == 61504.0

    with raises(KeyError):
        X = load_volfile('test_affine/test_data/atlas.npz')


def test_loader_age():
    age_atlas = np.load('test_affine/test_data/atlas.npz')['Age']
    atlas_list = ['test/atlas_00001.npz', 'test/atlas_00002.npz', 'test/atlas_00003.npz', 'test/atlas_00004.npz', 'test/atlas_00005.npz']
    gt_file = 'test_affine/test_data/test_data/00001_vol_annotation.npz'
    X = np.zeros((1,64,64,64,1))

    XX = loader_age(gt_file, 1, age_atlas, X, atlas_list)
    assert XX[0,0,0,0,0] == 2
    assert XX.shape == X.shape
    assert np.sum(XX) == 2

    XX = loader_age(gt_file, 3, age_atlas, X, atlas_list)
    assert XX[0,0,0,0,0] == 1
    assert XX[0,1,0,0,0] == 2
    assert XX[0,2,0,0,0] == 3
    assert np.sum(XX) == 6

    XX = loader_age(gt_file, 3, age_atlas, X, atlas_list, mode='give1')
   
    XX[0,0,0,0,0] = 0
    assert np.sum(XX) == 0.0

def test_apply_flips():
    img = nib.load('test_affine/test_data/test_data/00001_vol.nii.gz').get_fdata()
    T = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_t']
    B = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_b']
    
    img_aug, T_aug, B_aug = apply_flips(img, T, B, size = 32, print_flip = False, test = True,  test_lr = True, test_ud = False, test_fb = False)
    assert img_aug[T_aug[0],T_aug[1],T_aug[2]] == 1
    assert np.sum(T_aug-[47,17,32]) < 1e-5
    
    T = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_t']
    img_aug, T_aug, B_aug = apply_flips(img, T, B, size = 32, print_flip = False, test = True, test_lr = False, test_ud = True, test_fb = False)
    
    assert img_aug[T_aug[0],T_aug[1], T_aug[2]] == 1
    assert np.sum(T_aug - [17,47,32]) < 1e-5

    T = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_t']
    img_aug, T_aug, B_aug = apply_flips(img, T, B, size = 32, print_flip = False, test = True,  test_lr = False, test_ud = False,  test_fb = True)
    assert img_aug[T_aug[0], T_aug[1], T_aug[2]] == 1
    assert np.sum(T_aug - [47,47, 32]) < 1e-5

    T = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_t']
    img_aug, T_aug, B_aug = apply_flips(img, T, B, size = 32, print_flip = False, test = True, test_lr = True, test_ud = True, test_fb = False)
    assert img_aug[T_aug[0],T_aug[1], T_aug[2]] == 1
    assert np.sum(T_aug - [17, 17, 32]) < 1e-5

    img_aug, T_aug, B_aug = apply_flips(img, [], [], size = 32, print_flip = False, test = True, test_lr = True, test_ud = False, test_fb = False)
    assert img_aug[47,17,32] == 1
    
    T = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_t']
    img_aug, T_aug, B_aug = apply_flips(img, T, B, size = 32, print_flip = False, test = True, test_lr = False, test_ud = False, test_fb = False )
    assert np.sum(img_aug - img) < 1e-5
    assert np.sum(T_aug - [47,47,32]) < 1e-5


def test_apply_rot90():
    img = nib.load('test_affine/test_data/test_data/00001_vol.nii.gz').get_fdata()
    T = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_t']
    B = np.load('test_affine/test_data/test_data/00001_vol_annotation.npz')['coor_b']

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = True, test = True, r_test = 9)

    assert np.sum(img_aug - img) < 1e-5
    assert np.sum(T_aug - [47,47,32]) < 1e-5
    
    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = True, test = True, r_test = 0)

    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1    
    assert np.sum(T_aug - [17,47,32]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 1)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1    
    assert np.sum(T_aug - [47,17,32]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 2)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1    
    assert np.sum(T_aug - [47,32,47]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 3)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1
    assert np.sum(T_aug - [47,32,17]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 4)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1
    assert np.sum(T_aug - [32,47,17]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 5)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1
    assert np.sum(T_aug - [32,47,47]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 6)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1
    assert np.sum(T_aug - [47,17,32]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 7)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1
    assert np.sum(T_aug - [17,47,32]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, T, B, size = 32, print_rot90 = False, test = True, r_test = 8)
    assert img_aug[int(T_aug[0]),int(T_aug[1]),int(T_aug[2])] == 1
    assert np.sum(T_aug - [47,47,32]) < 1e-5

    img_aug, T_aug, B_aug = apply_rot90(img, [], [], size = 32, print_rot90 = False, test = True, r_test = 0)
    assert img_aug[17,47,32] == 1

def test_img_anno():
    X_orig = nib.load('test_affine/test_data/test_data/00001_vol.nii.gz').get_fdata()
    gt_file = 'test_affine/test_data/test_data/00001_vol_annotation.npz'

    X, Y = loader_img_anno(X_orig, gt_file, '1', [0,0])
    assert np.sum(X_orig - X) < 1e-5
    assert np.sum(Y[0,0:3]-[47,47,32]) < 1e-5

    X, Y = loader_img_anno(X_orig, gt_file,'1', [1,0], test=True)
    assert X[47,17,32] == 1
    assert np.sum(Y[0,0:3]-[47,17,32]) < 1e-5

    X, Y = loader_img_anno(X_orig, gt_file,'1', [0,1], test=True)
    assert X[17,47,32] == 1
    assert np.sum(Y[0,0:3]-[17,47,32]) < 1e-5

    X, Y = loader_img_anno(X_orig, gt_file,'1', [1,1], test = True, test_flips = True)
    assert X[47,17,32] == 1
    assert np.sum(Y[0,0:3]-[47,17,32]) < 1e-5

    X, Y = loader_img_anno(X_orig, gt_file,'1', [1,1], test=True, test_flips=False)
    assert X[17,47,32] == 1
    assert np.sum(Y[0,0:3]-[17,47,32]) < 1e-5

    X, Y = loader_img_anno(X_orig, gt_file, '2', [1,1])
    assert np.sum(Y) == 0

def test_image_generator():
    vol_names = glob.glob(os.path.join('test_affine/test_data/test_data', '*.nii.gz'))
    age_atlas = np.load('test_affine/test_data/atlas.npz')['Age']
    atlas_list = ['test/atlas_00001.npz', 'test/atlas_00002.npz', 'test/atlas_00003.npz', 'test/atlas_00004.npz', 'test/atlas_00005.npz']

    output = list(image_generator(vol_names, '1', 1, age_atlas, [0,0], atlas_list,  test=True))

    assert output[0][0].shape == (1,64,64,64,1)
    assert np.sum(output[0][0]) == 61504.0

    assert output[0][1].shape == (1,64,64,64,1)

    assert output[0][2].shape == (1,12)
    assert np.sum(output[0][2][0,0:3] - [47,47,32]) < 1e-5
    assert np.sum(output[0][2][0,3:6] - [40,60,32]) < 1e-5

    output = list(image_generator(vol_names, '2', 1, age_atlas, [0,0], atlas_list, test=True))

    assert np.sum(output[0][2]) == 0

    output = list(image_generator(vol_names, '1', 4, age_atlas, [0,0], atlas_list, test=True))

    assert np.sum(output[0][1]) >= 6

    output = list(image_generator(vol_names, '1', 1, age_atlas, [1,0], atlas_list, test=True))

    assert output[0][0].shape == (1,64,64,64,1)
    assert np.sum(output[0][0] - 61504) < 1e-5

    assert output[0][2].shape == (1,12)

def test_datagenerator_affine():
    vol_names = glob.glob(os.path.join('test_affine/test_data/test_data', '*.nii.gz'))
    age_atlas = np.load('test_affine/test_data/atlas.npz')['Age']
    atlas_list = ['test/atlas_00001.npz', 'test/atlas_00002.npz', 'test/atlas_00003.npz', 'test/atlas_00004.npz', 'test/atlas_00005.npz']

    train_image_gen =image_generator(vol_names, '1', 1, age_atlas, [0,0], atlas_list)
    output = list(datagenerator_affine(train_image_gen,test=True))

    assert output[0][0][0].shape == (1,64,64,64,1)
    assert np.sum(output[0][0][0]) == 61504.0
 
    assert output[0][1][0].shape == (1,64,64,64,1)
 
    assert output[0][1][1].shape == (1,12)
    assert np.sum(output[0][1][1][0,0:3] - [47,47,32]) < 1e-5
    assert np.sum(output[0][1][1][0,3:6] - [40,60,32]) < 1e-5

def test_select_index_atlas():
    indicator = [1, 1, 0, 0]

    idx = select_index_atlas(indicator)

    assert idx == 0 or idx == 1

    indicator = [1, 0, 0, 0]
    idx = select_index_atlas(indicator)

    assert idx == 0

    indicator = [0, 1, 1, 1]
    idx = select_index_atlas(indicator)
    assert idx != 0

def test_give_index_atlas():
    indicator = [1, 1, 0, 0]
    idx = give_index_atlas(indicator)

    assert idx[0] == 0 and idx[1] == 1
    assert len(idx) == 2

    indicator = [0, 0, 1, 0]
    idx = give_index_atlas(indicator)

    assert idx[0] == 2
    assert len(idx) == 1
