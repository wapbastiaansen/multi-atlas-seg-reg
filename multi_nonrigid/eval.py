import numpy as np
import nibabel as nib
import pandas as pd
import os
import glob
import tensorflow as tf
from neuron.layers import SpatialTransformer
import multi_affine.eval as ae
from multi_affine.utils import load_multi_atlas
from pandas import ExcelWriter
import SimpleITK as sitk

def eval_nonrigid(data_file, ev_file, atlas_dir, warp_dir, seg_dir, prepro_dir, atlas_list, subset, M, save_seg, diffeo=False):
    """
    Calculates:
    1. the dice overlap score between gt segmentations and segmentations after invers nonrigid deformation and inverse affine transofrmation
    2. the embryonic volume error between foudn EV in original space and EV measured in VR.

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

    Returns:
        Dice: List with dice scores per image, orderd as data_file
        EV: List with EV error per image, ordered as the data_file
        Updated data file
        summary files

    """

    # load needed files, initiate arrays for results
    data = pd.read_excel(data_file)
    data = data.iloc[:,0:4]
    ev = pd.read_excel(ev_file)
    Dice_nonrigid = []
    EV_nonrigid = []
    EV_AI = []
    EV_GT = []

    
    # load ga + segs atlases
    atlasses, segs, Age, atlas_files, A_t, A_b = load_multi_atlas(atlas_dir, atlas_list, False, True, True)

    # get list of all available gt segmentations
    aval_segs = os.listdir(seg_dir)

    # calculate measures for every image registered
    moving_image_name = []

    for i in range(len(data)):
        week = np.floor(data.iat[i,2]/7)

        file_name = os.path.basename(data.iat[i,0])
        print(file_name)
        idx = data.iat[i,3]

        # load the learned affine and nonrigid transformation + apply inverse to obtain segmentation in original image
        T = np.load(data.iat[i,0].split('_moved_affine')[0]+'_warp_affine.npy')
        if diffeo == True:
            phi_inv = np.load(warp_dir + '/' + file_name.split('.nii')[0]+'_warp_inv_nonrigid_'+ atlas_files[idx].split(atlas_dir +'/')[1].split('.npz')[0]+'.npy')
        else:
            phi = np.load(warp_dir + '/' + file_name.split('.nii')[0]+'_warp_nonrigid_'+atlas_files[idx].split(atlas_dir +'/')[1].split('.npz')[0]+'.npy')
            deform=phi[0,:,:,:,:]
            disp_field = sitk.GetImageFromArray(deform, isVector=True)
            deform = sitk.GetArrayFromImage(disp_field)
            phi_inv = sitk.GetArrayFromImage(sitk.InvertDisplacementField(disp_field))[np.newaxis,...]
        
        seg = apply_nonrigid_affine_inv(segs[idx,:,:,:,0], phi_inv, T, seg=True)
        
        
        file_seg_orig = data.iat[i,0].split('/'+subset+'/')[1].split('-')[0].split('_moved_affine')[0]+'_seg.nii.gz'

        # if the gt is available, calculate the DICE score, note dice GT is in the space after preprocessing.

        if M == 1:

            if save_seg == True:
                new_nifti = nib.Nifti1Image(seg, np.identity(4))
                nib.save(new_nifti, data_file.split('/outcome')[0] + '/seg_' + file_name.split('.nii')[0]+'_moved_nonrigid'+'.nii.gz')
                
            if file_seg_orig in aval_segs:
                Seg_orig = nib.load(seg_dir+'/'+file_seg_orig)
                Seg_orig = Seg_orig.get_fdata()
            
                dice = ae.calculate_dice(Seg_orig, seg)
                Dice_nonrigid.append(dice)
            else:
                Dice_nonrigid.append(np.nan)
        

            # if the EV is available, calculate the embryonic volume error, EV_VR is measured in original image, hence we have to compensate for our preprocessing.
            df = ev.loc[ev['ID'] == int(data.iat[i,1])]
            if len(df)>0:
                found = False
                for j in range(len(df)):
                    if np.floor(df.iat[j,1]/7) == week:
                        EV_VR = df.iat[j,2]
                        found = True

                if found == False:
                    EV_nonrigid.append(np.nan)
                    EV_AI.append(np.nan)
                    EV_GT.append(np.nan)
                else:
                    vox_size = nib.load(prepro_dir.split('/prepro')[0]+'/' + subset +'/' +file_name.split('_moved_affine')[0]+'.nii.gz').header['pixdim'][1]
                    factor = np.load(prepro_dir+'/' + data.iat[i,0].split('/'+subset+'/')[1].split('_moved_affine')[0]+'_preprocess.npz')['zoom_factor']
                    ev_error, EV_ai = ae.calculate_ev_error(seg, EV_VR, vox_size, factor)
                    EV_nonrigid.append(ev_error)
                    EV_AI.append(EV_ai)
                    EV_GT.append(EV_VR)
            else:
                EV_nonrigid.append(np.nan)
                EV_AI.append(np.nan)
                EV_GT.append(np.nan)


        else:
            #M>1
            if file_name.split('_moved')[0] == moving_image_name:
                count+=1
                if count == M:
                    #add last segmentation and do majority voting
                    seg_multi = np.concatenate((seg_multi, seg[np.newaxis, ...]),axis=0)

                    seg_major = ae.majority_voting(seg_multi)
                    if save_seg == True:
                        new_nifti = nib.Nifti1Image(seg_major, np.identity(4))
                        nib.save(new_nifti, data_file.split('/outcome')[0] + '/seg_' + file_name.split('.nii')[0]+'_moved_nonrigid'+'.nii.gz')
                        
                
                    if file_seg_orig in aval_segs:
                        Seg_orig = nib.load(seg_dir+'/'+file_seg_orig)
                        Seg_orig = Seg_orig.get_fdata()

                        dice = ae.calculate_dice(Seg_orig, seg_major)
                        Dice_nonrigid.append(dice)
                    else:
                        Dice_nonrigid.append(np.nan)
 
                
                    # if the EV is available, calculate the embryonic volume error, EV_VR is measured in original image, hence we have to compensate for our preprocessing.
                    df = ev.loc[ev['Predictnr'] == int(data.iat[i,1])]
                    if len(df)>0:
                        found = False
                        for j in range(len(df)):
                            if np.floor(df.iat[j,1]/7) == week:
                                EV_VR = df.iat[j,2]
                                found = True
 
                        if found == False:
                            EV_nonrigid.append(np.nan)
                            EV_GT.append(np.nan)
                            EV_AI.append(np.nan)
                        else:
                            vox_size = nib.load(prepro_dir.split('/prepro')[0]+'/' + subset +'/' +file_name.split('_moved_affine')[0]+'.nii.gz').header['pixdim'][1]
                            factor = np.load(prepro_dir+'/' + data.iat[i,0].split('/'+subset+'/')[1].split('_moved_affine')[0]+'_preprocess.npz')['zoom_factor']
                            ev_error, ev_ai = ae.calculate_ev_error(seg, EV_VR, vox_size, factor)
                            EV_nonrigid.append(ev_error)
                            EV_GT.append(EV_VR)
                            EV_AI.append(ev_ai)

                    else:
                        EV_nonrigid.append(np.nan)
                        EV_GT.append(np.nan)
                        EV_AI.append(np.nan)
                

                else:
                    #add segmentation
                    seg_multi = np.concatenate((seg_multi, seg[np.newaxis, ...]),axis=0)
                    EV_nonrigid.append(np.nan)
                    EV_GT.append(np.nan)
                    EV_AI.append(np.nan)
                    Dice_nonrigid.append(np.nan)
            else:
                #we arrived at new image: restart collecting segmentations
                count = 1
                seg_multi = seg[np.newaxis,...]
                moving_image_name = file_name.split('_moved')[0]
                EV_nonrigid.append(np.nan)
                EV_GT.append(np.nan)
                EV_AI.append(np.nan)
                Dice_nonrigid.append(np.nan)
 
    data['Dice'] = Dice_nonrigid
    data['Ev'] = EV_nonrigid
    data['Ev_GT'] = EV_GT
    data['Ev_AI'] = EV_AI

    
    writer = ExcelWriter(data_file)
    data.to_excel(writer, index=False)
    writer.save()

    summarize_results_nonrigid(data,data_file.split('/outcome')[0])
    
    return Dice_nonrigid, EV_nonrigid

def summarize_results_nonrigid(dataframe, save_dir):
    """
    Function to create .txt file containing a summary of the results.
 
    Args:
        dataframe: the dataframe with performance measures per image
    	save_dir: directory to save the .txt file
 
    Returns:
        .txt file giving the mean EV error, Dice score, +/- std
    
    """

    mean_dice = dataframe['Dice'].mean()
    std_dice = dataframe['Dice'].std()
 
    mean_ev = dataframe['Ev'].mean()
    std_ev = dataframe['Ev'].std()
  
    text_file = open(save_dir + '/summary_results.txt', 'w+')
    text_file.write('EV error: ' +str(mean_ev) +'+/-' +str(std_ev) +'\n')
    text_file.write('Dice: ' +str(mean_dice) +'+/-' +str(std_dice) +'\n')

def apply_nonrigid_affine_inv(img, phi_inv, T, seg=True):
    """
    Applies the inverse nonrigid and affine transformation to image img.
    Note that img can be a segmentation.
     
    Args:
        T: affine transformation, output of Voxelmorph, size (1,12)
        phi_inv: inverse nonrigid deformation
        img: image to be transformed
        seg: true if image is a segmentation, then output image is a segmenation

    Returns:
    Y: image/segmentation after applying the inverse nonrigid and affine deformation
    """

    tf.enable_eager_execution()
    
    img_phi = SpatialTransformer(interp_method='linear', indexing='ij')([tf.cast(img[np.newaxis,...,np.newaxis], tf.float32), tf.cast(phi_inv, tf.float32)])

    assert(T.shape == (1,12))
    TT = np.zeros((4,4))
    TT[0,:] = T[0,0:4]
    TT[1,:] = T[0,4:8]
    TT[2,:] = T[0,8:]
 
    TT += np.identity(4)
    T_inv = np.linalg.inv(TT)
    T_inv += -np.identity(4)
    TT_inv = np.reshape(np.concatenate([T_inv[0,0:4],T_inv[1,0:4],T_inv[2,0:4]]),[1,12])

    Y = SpatialTransformer(interp_method='linear', indexing='ij')(    [img_phi,tf.cast(TT_inv,tf.float32)])

    Y = Y.numpy()
    Y = Y[0,:,:,:,0]
    if seg == True:
        Y[Y>=0.5] = 1
        Y[Y<0.5] = 0

    return Y

def apply_nonrigid_affine(img, phi, T, seg=True):
    """
    Applies the nonrigid and affine transformation to image img.
    Note that img can be a segmentation.
     
    Args:
        T: affine transformation, output of Voxelmorph, size (1,12)
        phi: nonrigid deformation
        img: image to be transformed
        seg: true if image is a segmentation, then output image is a segmenation

    Returns:
    Y: image/segmentation after applying the nonrigid and affine deformation
    """

    tf.enable_eager_execution()
    
    img_T = SpatialTransformer(interp_method='linear', indexing='ij')([tf.cast(img[np.newaxis,...,np.newaxis], tf.float32), tf.cast(T, tf.float32)])

    Y = SpatialTransformer(interp_method='linear', indexing='ij')([img_T,tf.cast(phi,tf.float32)])

    Y = Y.numpy()
    Y = Y[0,:,:,:,0]
    if seg == True:
        Y[Y>=0.5] = 1
        Y[Y<0.5] = 0

    return Y
