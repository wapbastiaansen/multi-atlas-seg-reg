import numpy as np
import nibabel as nib
import pandas as pd
import os
import tensorflow as tf
import math
from neuron.layers import SpatialTransformer
from multi_affine.datagenerators import indicator, give_index_atlas
from multi_affine.utils import load_multi_atlas
from pandas import ExcelWriter

def eval_affine(data_file, ev_file,  atlas_dir, warp_dir, seg_dir, prepro_dir, atlas_list,  subset, M, save_seg, print_output=True):
    """
    Calculates:
    1. the dice overlap score between ground truth segmentations 
    (in original space) and segmentations obtained after inverse affine
    transformation for a given file.
    2. the embryonic volume error between found EV in original space and EV measured in VR.
    
    Args:
        data_file: excel file containing images for which we will calculate the Dice and EV error (if avalaible)
        ev_file: excel file containing all the EV_VR
        atlas_dir: directory where eligble atlas files can be found
        warp_dir: directory where used affine transformations are saved
        seg_dir: directory containing ground truth segmentations
        prepro_dir: directory containing the prepro .npz files
        subset: set of the data for which we calculate Dice and EV error
        atlas_list: list of elibigle atlases
        M: number of eligible atlases used for training, if M>1 we use majority voting
        save_seg: if segmentations are saved

    Returns:
        Dice_affine: List with dice scores per image, orderd as data_file
        EV_affine: List with EV error per image, ordered as the data_file
        Updated data file
    """

    # load needed files, initiate arrays for results
    data = pd.read_excel(data_file)
    data = data.iloc[:,0:4]
    ev = pd.read_excel(ev_file)
    Dice_affine = []
    EV_affine = []
    EV_GT = []
    EV_AI = []
    # load ga + segs atlases
    atlasses, segs, Age, atlas_files, A_t, A_b = load_multi_atlas(atlas_dir, atlas_list, False, True, True)

    # get list of all available gt segmentations
    aval_segs = os.listdir(seg_dir)

    # calculate measures for every image registered
    for i in range(len(data)):
        #get indicator + weeknr to select right atlas for comparision
        indi = indicator(data.iat[i,2],M,Age, atlas_files)
        idx = give_index_atlas(indi)
        
        week = np.floor(data.iat[i,2]/7)
    
        file_name = os.path.basename(data.iat[i,0])

        # load the learned affine transformation + apply inverse to obtain segmentation in original image
        T = np.load(warp_dir+'/'+file_name.split('.nii')[0]+'_warp_affine.npy')
        
        for j in range(M):
 
            seg = apply_affine_inv(T, segs[idx[j],:,:,:,0], seg=True)
            if j == 0:
                Seg_affine_inv = seg[np.newaxis, ...]
            else:
                Seg_affine_inv = np.concatenate((Seg_affine_inv, seg[np.newaxis, ...]),axis=0)
        
        if M>1:
            Seg_affine = majority_voting(Seg_affine_inv)
        else:
            Seg_affine = Seg_affine_inv[0,:,:,:]

        if save_seg == True:
            new_nifti = nib.Nifti1Image(Seg_affine, np.identity(4))
            nib.save(new_nifti, data_file.split('/outcome')[0] + '/seg_' + file_name.split('.nii')[0]+'_moved_nonrigid_'+atlas_files[    j].split(atlas_dir +'/')[1].split('.npz')[0]+'.nii.gz')


        file_seg_orig = data.iat[i,0].split('/'+subset+'/')[1].split('-')[0].split('.nii')[0]+'_seg.nii.gz'
 
        # if the gt is available, calculate the DICE score, note dice GT is in the space after preprocessing.

        if file_seg_orig in aval_segs:    
            Seg_orig = nib.load(seg_dir+'/'+file_seg_orig)
            Seg_orig = Seg_orig.get_fdata()
        
            dice = calculate_dice(Seg_orig, Seg_affine)
            Dice_affine.append(dice)
        else:
            Dice_affine.append(np.nan)
    
        Dice = [data.iat[i,0]]
        if M>1:
            if file_seg_orig in aval_segs:
                Seg_orig = nib.load(seg_dir+'/'+file_seg_orig)
                Seg_orig = Seg_orig.get_fdata()
                m = 0
                for j in range(len(atlas_files)):
                    if indi[j] == 1:
                        seg = Seg_affine_inv[m,:,:,:]
                        m+=1
                        dice = calculate_dice(Seg_orig, seg)
                        Dice.append(dice)
                    else:
                        Dice.append(np.nan)
                         
                print(m)
                assert(m==M)
                
            else:
                for j in range(len(atlas_files)):
                    Dice.append(np.nan)

        # if the EV is available, calculate the embryonic volume error, EV_VR is measured in original image, hence we have to compensate for our preprocessing.
        df = ev.loc[ev['ID'] == int(data.iat[i,1])]
        if len(df)>0:
            found = False
            for j in range(len(df)):
                if np.floor(df.iat[j,1]/7) == week:
                    EV_VR = df.iat[j,2]
                    found = True

            if found == False:
                EV_affine.append(np.nan)
                EV_GT.append(np.nan)
                EV_AI.append(np.nan)
            else:
                vox_size = nib.load(data.iat[i,0]).header['pixdim'][1]
                factor = np.load(prepro_dir+'/' + data.iat[i,0].split('/'+subset+'/')[1].split('.nii')[0]+'_preprocess.npz')['zoom_factor']
                ev_error, ev_AI = calculate_ev_error(Seg_affine, EV_VR, vox_size, factor)
                EV_affine.append(ev_error)
                EV_GT.append(EV_VR)
                EV_AI.append(ev_AI)
        else:
            EV_affine.append(np.nan)
            EV_GT.append(np.nan)
            EV_AI.append(np.nan)
        
        EV = [data.iat[i,0]]
        if M>1 and found == True:
            m=0
            for j in range(len(atlas_files)):
                if indi[j] == 1:
                    seg = Seg_affine_inv[m,:,:,:]
                    m+=1
                    ev_error = calculate_ev_error(seg, EV_VR, vox_size, factor)
                    EV.append(ev_error)
                else:
                    EV.append(np.nan)
            
            assert(m==M)
 
        else:
            for j in range(len(atlas_files)):
                EV.append(np.nan)

    data['Dice'] = Dice_affine
    data['Ev'] = EV_affine
    data['Ev_GT'] = EV_GT
    data['Ev_AI'] = EV_AI

    writer = ExcelWriter(data_file)
    data.to_excel(writer, index=False)
    writer.save()

    summarize_results(data,data_file.split('/outcome')[0])
  
    return Dice_affine, EV_affine


def apply_affine_inv(T, img, seg=False):
    """
    Applies the inverse affine transformation of transformation T to image img.
    Note that img can be a segmentation.
    
    Args:
        T: affine transformation, output of Voxelmorph, size (1,12)
        img: image to be transformed
        seg: true if image is a segmentation, then output image is a segmenation
    
    Returns:
        Y: image/segmentation after applying the inverse affine transform of T
    """
    tf.enable_eager_execution()
    assert(T.shape == (1,12))
    TT = np.zeros((4,4))
    TT[0,:] = T[0,0:4]
    TT[1,:] = T[0,4:8]
    TT[2,:] = T[0,8:]
        
    TT += np.identity(4)
    T_inv = np.linalg.inv(TT)
    T_inv += -np.identity(4)
    TT_inv = np.reshape(np.concatenate([T_inv[0,0:4],T_inv[1,0:4],T_inv[2,0:4]]),[1,12])
       
    Y = SpatialTransformer(interp_method='linear', indexing='ij')([tf.cast(img[np.newaxis,...,np.newaxis],tf.float32),tf.cast(TT_inv,tf.float32)])
    
    Y = Y.numpy()
    Y = Y[0,:,:,:,0]
    if seg == True:
        Y[Y>=0.5] = 1
        Y[Y<0.5] = 0
    
    return Y

def calculate_dice(seg1, seg2):
    """
    Calculate DICE overlap score between seg1 and seg2.
    
    Args:
        seg1: ground truth segmentation.
        seg2: segmentation to compare with seg1
        
    Returns:
        dice overlap score
    """
    
    seg1 = np.asarray(seg1).astype(np.bool)
    seg2 = np.asarray(seg2).astype(np.bool)
            
    # Compute Dice coefficient
    intersection = np.logical_and(seg1, seg2)
    
    dice = 2. * intersection.sum() / (seg1.sum() + seg2.sum())
    
    return dice
    
def calculate_ev_error(Seg_affine_inv, EV_VR, vox_size, factor):
    """
    Function to calculate the error in embryonic volume between found seg_affine_inv en EV_VR (gt measured in VR).

    Args:
        Seg_affine_inv: found segmentation in original space
        EV_VR: gt measured in VR
        vox_size: voxel size of original image
        factor: zooming factor used during preprocessing

    Returns:
        Relative error in embryonic volume
    """
    vox_size = vox_size*4*(1/factor)
    EV_AI = vox_size*vox_size*vox_size*np.sum(Seg_affine_inv)*(1/1000)
    EV_AI=np.array(EV_AI)
    EV_VR=np.array(EV_VR)

    return np.abs(((EV_AI-EV_VR)/EV_VR)), EV_AI

def summarize_results(dataframe, save_dir):
    """
    Function to create .txt file containing a summary of the results.

    Args:
        dataframe: the dataframe with performance measures per image
        save_dir: directory to save the .txt file

    Returns:
        .txt file giving the mean landmark error, EV error, Dice score, +/- std
        Note: mean landmark error is in mm: voxelsize in atlas space 0.62 mm
    """
    mean_dice = dataframe['Dice'].mean()
    std_dice = dataframe['Dice'].std()

    mean_ev = dataframe['Ev'].mean()
    std_ev = dataframe['Ev'].std()

    mean_le = dataframe[3].mean()*0.62
    std_le = dataframe[3].std()*0.62

    
    text_file = open(save_dir + '/summary_results.txt', 'w+')
    text_file.write('landmark error: ' +str(mean_le) +'+/-' +str(std_le)+'\n')
    text_file.write('EV error: ' +str(mean_ev) +'+/-' +str(std_ev)+'\n')
    text_file.write('Dice: ' +str(mean_dice) +'+/-' +str(std_dice)+'\n')
    
def majority_voting(seg):
    """
    Function that gives segmentation that is results of majority voting of the N segmentations in seg.

    Args:
        seg: Nxdims numpy array with resulting segmentations

    Returns:
        major_seg: resulting segmentation after majority voting

    """

    majority_limit = seg.shape[0]/2

    votes = np.sum(seg, axis=0)

    votes[votes <= majority_limit] = 0
    votes[votes > majority_limit] = 1

    major_seg = votes

    return major_seg
