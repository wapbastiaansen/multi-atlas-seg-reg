from multi_nonrigid.datagenerators import *
import numpy as np
import glob
import os

def test_select_index_atlas():
    indi = [1, 0, 0]

    index_select = select_index_atlas(indi)

    assert index_select == 0

    indi = [1, 1, 0]

    index_select = select_index_atlas(indi)

    assert index_select != 2

def test_image_generator():
    vol_names = glob.glob(os.path.join('test_nonrigid/test_data', '*.nii.gz'))
    age_atlas = np.load('test_nonrigid/test_data/atlas.npz')['Age']

    atlasses = np.load('test_nonrigid/test_data/atlas.npz')['atlas_vol']
    atlas_list = ['test/atlas_00001.npz', 'test/atlas_00002.npz', 'test/atlas_00003.npz', 'test/atlas_00001.npz', 'test/atlas_00002.npz']
    
    output = list(image_generator(vol_names, 1, atlasses, age_atlas, atlas_list, test=True))

    assert output[0][0].shape == (1,64,64,64,1)
    assert np.sum(output[0][0]) == 61504.0

    assert output[0][1].shape == (1,64,64,64,1)
    assert np.sum(output[0][1]) == 0.0

def test_datagenerator_nonrigid():
    vol_names = glob.glob(os.path.join('test_nonrigid/test_data', '*.nii.gz'))
    age_atlas = np.load('test_nonrigid/test_data/atlas.npz')['Age']
    atlasses = np.load('test_nonrigid/test_data/atlas.npz')['atlas_vol']
    atlas_list = ['test/atlas_00001.npz', 'test/atlas_00002.npz', 'test/atlas_00003.npz', 'test/atlas_00001.npz', 'test/atlas_00002.    npz']


    gen = image_generator(vol_names, 1, atlasses, age_atlas, atlas_list, test=True)
    
    output = list(datagenerator_nonrigid(gen))

    assert output[0][0][0].shape == (1,64,64,64,1)
    assert np.sum(output[0][0][0]) == 61504.0

    assert output[0][0][1].shape == (1,64,64,64,1)
    assert np.sum(output[0][0][1]) == 0.0

    assert output[0][1][0].shape == (1,64,64,64,1)
    assert np.sum(output[0][1][0]) == 0.0

    assert output[0][1][1].shape == (1,64,64,64,3)
    assert np.sum(output[0][1][1]) == 0.0

    gen = image_generator(vol_names, 1, atlasses, age_atlas, atlas_list, test=True)
    output = list(datagenerator_nonrigid(gen, diffeomorphic=True))

    assert output[0][1][2].shape == (1,64,64,64,3)
    assert np.sum(output[0][1][2]) == 0.0
