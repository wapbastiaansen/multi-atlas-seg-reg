from multi_affine.utils import * 
import glob
import os
import matplotlib.pyplot as plt

def test_load_multi_atlas():
    atlas_dir=os.getcwd()+'/test_affine/test_data/atlas_dir' 
    atlas_list = ['00001']
    atlasses, segs, Age, atlas_files, A_t, A_b = load_multi_atlas(atlas_dir, atlas_list, True, True, True)
    
    assert atlasses.shape == (5,64,64,64,1)
    assert segs.shape == (5,64,64,64,1)
    assert Age.shape == (5,1)
    assert np.sum(A_t - [1,1,1]) < 1e-5
    assert np.sum(A_b - [0,0,0]) < 1e-5
    assert atlas_files[0] == os.getcwd()+'/test_affine/test_data/atlas_dir/atlas_00001_8.npz'
    assert atlas_files[4] == os.getcwd()+'/test_affine/test_data/atlas_dir/atlas_00001_12.npz'
    
    atlasses, segs, Age, atlas_files, A_t, A_b  = load_multi_atlas(atlas_dir, atlas_list, False, False, False)
    
    assert atlasses == []
    assert Age == []
    assert segs == []

    atlas_dir=os.getcwd()+'/test_affine/test_data/atlas_dir'
    atlas_list = ['00001', '00002']
    atlasses, segs, Age, atlas_files, A_t, A_b = load_multi_atlas(atlas_dir, atlas_list, True, True, True)
    assert atlasses.shape == (10,64,64,64,1)
    assert Age.shape == (10,1)
    #atlas 1 is always one day younger then atlas 2: check if sorting is correct
    assert atlas_files[0] == os.getcwd()+'/test_affine/test_data/atlas_dir/atlas_00001_8.npz'
    assert atlas_files[9] == os.getcwd()+'/test_affine/test_data/atlas_dir/atlas_00002_12.npz' 


def test_summary_experiment():
    directory = 'test_affine/test_data'
    parameter_values = [1, [1,0],'/test/file.nii',0.001,'47']
    parameter_names =['integer', 'list' , 'directory', 'learning_rate' , 'str_num']

    summary_experiment(directory, parameter_values, parameter_names)
    assert len(parameter_values) == len(parameter_names)

def test_get_predict_nr():
    file_ext = 'test_affine/test_data/00001_vol.nii.gz'

    predictnr = get_predict_nr(file_ext)

    assert predictnr == '00001'
