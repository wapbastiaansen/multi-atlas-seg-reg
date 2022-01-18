import os
import glob

import numpy as np
import nibabel as nib
import pandas as pd
from pandas import ExcelWriter

from multi_nonrigid import networks
import multi_affine.datagenerators as ad
import multi_affine.register as ar
from multi_affine.utils import load_multi_atlas, select_on_GA, get_predict_nr

def register(save_img,
             save_warp,
             save_warp_inv,
             save_visual,
             num,
             week_nr,
             data_dir,
             atlas_dir,
             save_dir,
             load_model_file,
             atlas_list,
             M,
             enc,
             dec,
             diffeomorphic=False,
             int_steps=7,
             start=0):

    """
    Function to register images using a trained network.

    Args:
        save_img: if set to true the moved image will be saved as nifti
        save_warp: if set to true the nonrigid deformation will be saved as npy file
        save_warp_inv: if set to true the inverse nonrigid deformation will be save as npy file
        save_visual: if set to true png with visualization of results will be saved
        num: number of images to register
        week_nr: if set to 'all' all data is register, otherwise only of the chosen week
        data_dir: directory of data that must be registered
        atlas_dir: directory with all atlas files
        save_dir: directory to save results
        load_model_file: weight file to load after training
        atlas_list: list of atlases used to train this model
        M: number of atlases used for optimization
        enc: filters in enc of trained model
        dec: fitlers of dec of trained model
        diffeomorhpic: if set to true the diffeomorphic model will be evaluated
        int_steps: number of int steps used for training diffeomorphic model
        start: start with registering somewhere in the list, default is 0

    Returns:
        data: numpy array wiht in each row of the file, predictnr and GA.

    """

    # load atlas and metadata from provided files.
    atlasses, segs, Age, atlas_files, A_t, A_b  = load_multi_atlas(atlas_dir, atlas_list, True, True, False)
    
    vol_size = atlasses.shape[1:-1]
    
    test_vol_names = sorted(glob.glob(os.path.join(data_dir, '*.nii.gz')))
    if start != 0:
        test_vol_names = test_vol_names[start:]

    if week_nr!='all':
        test_vol_names = select_on_GA(test_vol_names, week_nr)

    assert len(test_vol_names) > 0, "Could not find any test data"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # load model
    model = networks.network_nonrigid(vol_size,enc,dec,diffeomorphic=diffeomorphic, int_steps=int_steps)

    model.load_weights(load_model_file)

    data = []
    im_results = []
    labels = []

    for i in range(num):
        img = test_vol_names[i]
        gt_file = test_vol_names[i].split('_moved_affine')[0]+'_annotation.npz'
        age = int(ad.load_volfile(gt_file,np_var='GA'))
        indi = ad.indicator(age, M, Age, atlas_files)
        
        for j in range(len(indi)):
            if indi[j] == 1:
                out, mov, phi, phi_inv = register_image_nonrigid(img, atlasses[j,:,:,:,:][np.newaxis,...], age, model, diffeomorphic)

                if save_img == True:
                    new_nifti = nib.Nifti1Image(mov, np.identity(4))
                    nib.save(new_nifti, save_dir + '/' + img.split(data_dir +'/')[1].split('.nii')[0]+'_moved_nonrigid_'+atlas_files[j].split(atlas_dir +'/')[1].split('.npz')[0]+'.nii.gz')
                if save_warp == True:
                    np.save(save_dir+'/'+img.split(data_dir+'/')[1].split('.nii')[0]+'_warp_nonrigid_'+atlas_files[j].split(atlas_dir +'/')[1].split('.npz')[0], phi)
                if save_warp_inv == True and diffeomorphic == True:                 
                    np.save(save_dir+'/'+img.split(data_dir+'/')[1].split('.nii')[0]+'_warp_inv_nonrigid_'+atlas_files[j].split(atlas_dir +'/')[1].split('.npz')[0], phi_inv)

                if save_visual == True:
                    im_results.append(mov[0,:,:,:,0])
                    labels.append([int(age), atlas_files[j].split(atlas_dir+'/')[1].split('.npz')[0].split('atlas_')[1].split('_')[0]])
                out.append(j)
                data.append(out)
    
    if save_visual == True:
        ar.save_png(im_results, labels, save_dir, 32, 64, 64, axis=2)
        ar.save_png(im_results, labels, save_dir, 32, 64, 64, axis=0)
        ar.save_png(im_results, labels, save_dir, 32, 64, 64, axis=1)

        
    Data = pd.DataFrame(data)
    writer = ExcelWriter(save_dir + '/outcome.xlsx')
    Data.to_excel(writer, index=False)
    writer.save()

    return data


def register_image_nonrigid(img, atlas, age, model, diffeomorphic=False):
    """
    Function to register nonrigidly one image.

    Args:
        img: img to be registered
        atlas: atlas to register image to
        model: model with loaded weights to register with
        diffeomorphic: if true also the inverse deformation is given

    Returns:
        out: list containing file_name, predictnr, GA
        mov: moved iamge
        phi: nonrigid deformation
        phi_inv: if diffeomorphic model: phi_inv
    """
    out = []
    X_vol = ad.load_volfile(img)[np.newaxis, ..., np.newaxis]

    predictnr = get_predict_nr(img)

    if diffeomorphic == True:
        [mov, phi, phi_inv] = model.predict([X_vol, atlas])
    else:
        [mov, phi] = model.predict([X_vol, atlas])
        phi_inv = []

    out.append(img)
    out.append(predictnr)
    out.append(age)
    
    return out, mov, phi, phi_inv
         

