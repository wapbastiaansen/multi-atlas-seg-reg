import numpy as np
import glob
import os
import nibabel as nib
from shutil import copyfile

def load_multi_atlas(atlas_dir, atlas_list, output_atlas, output_age, output_seg):
    """
    Loads for given directory all atlas files, their segmentations and
    corresponding GA in the same order.
    
    Args:
        atlas_dir: directory with all atlas images
        atlas_list: list of predictnr of atlases to include
        output_atlas: output atlas images
        output_age: output GA
        output_seg: output segmentations
        
    Returns:
        atlasses: numpy array containing all atlases
        Age: list containing all GA
        segs: numpy array containing all segmenations
        atlas_files: list of filenames
        A_t: ground truth top landmark
        A_b: ground truth bottom landmark
    """
    atlas_files = glob.glob(os.path.join(atlas_dir, '*.npz'))
   
    atlas_files, Age = sort_and_select_atlas(atlas_files, atlas_list, output_age)
     
    i=0
    
    for file in atlas_files:
        i+=1
        if output_atlas == True:
            atlas_vol = np.load(file)['vol'][np.newaxis, ..., np.newaxis]
            atlas_vol=atlas_vol.astype('float32')
            if i==1:
                atlasses=atlas_vol
            else:
                atlasses=np.concatenate([atlasses,atlas_vol],axis=0)
        else:
            atlasses=[]
            
        if output_seg == True:
            seg_vol = nib.load(atlas_dir+'/seg/seg_'+file.split(atlas_dir+'/atlas_')[1].split('.npz')[0]+'.nii.gz').get_fdata()[np.newaxis,...,np.newaxis]
            if i==1:
                segs=seg_vol
            else:
                segs=np.concatenate([segs,seg_vol],axis=0)
        else:
            segs=[]
    
    A_t = np.load(atlas_dir+'/landmark/ground_truth_landmark.npz')['A_t']
    A_b = np.load(atlas_dir+'/landmark/ground_truth_landmark.npz')['A_b']

    return atlasses, segs, Age, atlas_files, A_t, A_b

def select_on_GA(vol_names,week_nr):
    """
    Selects from list of files the files with the right week number.
    
    Args:
        vol_names: list with all files
        week_nr: the week number we wish to select
        
    Returns:
        matching: list with selected images
    """
    matching = [s for s in vol_names if '_US_'+week_nr in s]
 
    return matching
    
    
def summary_experiment(directory,parameter_values, parameter_names):
    """
    Function that creates a text file with information about an experiment.

    Args:
        directory: directory where to save the summary
        parameter_values: values of the parameters used for the experiment
        parameter_names: names of the parameters summarized

    Returns:
        summary_experiment.txt file with format: parameter_name[i]: parameter_value[i] *new_line*
    """
    assert len(parameter_values) == len(parameter_names)
    params = create_dictionairy(parameter_values, parameter_names)

    text_file = open(directory + '/summary_experiment.txt','w+')
    for var in params:
        text_file.write(str(var) + ': ' + str(params[var]) + '\n')
        
        
def create_dictionairy(variables,names):
    """
    Function to create a dictionary with as keys: names and as values: variables.
    """
    params = {}
    
    for i in range(0,len(variables)):
        params[names[i]] = variables[i]
        
    return params

def get_predict_nr(file_ext):
    """
    Function to get the predictnumber out of a file name.
    """
    name = os.path.basename(file_ext)
    if 'atlas' not in name:
        predictnr = name[:5]
    else:
        predictnr = name.split('atlas_')[1][:5]
    return predictnr

def sort_and_select_atlas(atlas_files, atlas_list, output_age):
    """
    Function to sort list of atlas files based on GA and select based on atlas_list.

    Args:
        atlas_files: list of all available atlases
        atlas_list: list of predict numbers we will use
        output_age: bool variable if we will output the age

    Returns:
        atlas_files: list of all select atlases that is sorted based on GA
        Age: (n,1) np array with gestational ages, sorted.
    """
    Age = []
    select_files = []
    for file in atlas_files:
        predictnr = get_predict_nr(file)
       
        if predictnr in atlas_list:
            age=np.load(file)['GA']
            Age.append(int(age))
            select_files.append(file)

    sort_index=np.argsort(Age)
    Age = np.take_along_axis(np.array(Age), sort_index, axis=0)
    Age = Age.reshape((len(Age),1))
    atlas_files = list(np.take_along_axis(np.array(select_files), sort_index, axis=0))

    if output_age == False:
        Age = []

    return atlas_files, Age

def copy_anno_files(old_dir, new_dir):
    """
    Function to copy annotation files from old_dir to new_dir
    """
    files = os.listdir(old_dir)

    for file in files:
        if '_annotation.npz' in file:
            copyfile(old_dir +'/' + file, new_dir +'/' + file)

