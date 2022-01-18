# python imports
import os
import glob

# third-party imports
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import pandas as pd
from pandas import ExcelWriter

# project imports
from multi_affine import datagenerators
from multi_affine import networks
from multi_affine.utils import load_multi_atlas, select_on_GA, get_predict_nr


def register(save_img,
             save_warp,
             save_visual,
             num,
             week_nr,
             data_dir,
             atlas_dir, 
             save_dir,
             load_model_file,
             filters_enc,
             num_fcl=4,
             nhidden=1000):
    """
    Function to register images using a trained network.

    Args:
        save_img: if set to true the moved image will be saved as nifti
        save_warp: if set to true the affine transformation matrix T will be saved as npy file
        save_visual: if set to true png with visualization of results will be saved
        num: number of images to register
        week_nr: if set to 'all' all data is register, otherwise only of the chosen week
        data_dir: directory of data that must be registered
        atlas_dir: directory with all atlas files
        save_dir: directory to save results
        load_model_file: weight file to load after training
        filters_enc: filters of encoder used for training
        num_fcl: number of fcl
        nhidden: number of neurons in fcl
        Note: filters_enc, num_fcl, nhidden must match the .h5 file!!

    Returns:
        data: numpy array with in each row the file, predictnr, GA and landmark error.
    """

    # load atlas and metadata from provided files.
    atlasses, segs, Age, atlas_files, A_t, A_b  = load_multi_atlas(atlas_dir, ['00001'], True, True, False)
    vol_size = atlasses.shape[1:-1] 
    test_vol_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    
    if week_nr!='all':
        test_vol_names = select_on_GA(test_vol_names, week_nr)
    
    assert len(test_vol_names) > 0, "Could not find any test data"
	
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # load model
    model=networks.network_multi_affine(vol_size,filters_enc,num_fcl,nhidden)
        
    model.load_weights(load_model_file)
    
    data = []
    im_result = []
    labels = []
    for i in range(num):
        img = test_vol_names[i]
        out, mov, T = register_image(img, A_t, A_b, model)
        if save_img == True: 
            new_nifti=nib.Nifti1Image(mov[0,:,:,:,0],np.identity(4))
            nib.save(new_nifti, save_dir +'/' + img.split(data_dir+ '/')[1].split('.nii')[0]+'_moved_affine.nii.gz')
        if save_warp == True:
            np.save(save_dir + '/'+ img.split(data_dir + '/')[1].split('.nii')[0] + '_warp_affine', T)
        if save_visual == True:
            im_result.append(mov[0,:,:,:,0])
            labels.append(int(out[2]))

        data.append(out)

    if save_visual == True:
        save_png(im_result, labels, save_dir, 32, 64, 64, axis=2)
        save_png(im_result, labels, save_dir, 32, 64, 64, axis=0)
        save_png(im_result, labels, save_dir, 32, 64, 64, axis=1)

    Data = pd.DataFrame(data)
    writer = ExcelWriter(save_dir + '/outcome.xlsx')
    Data.to_excel(writer, index=False)
    writer.save()

    return data

def register_image(img, A_t, A_b, model):
    """
    function to register one image.

    Args:
        img: img to be registered
        A_t: top landmark coordinate in atlas
        A_b: bottum landmark coordinate in atlas
        model: model with loaded weights to register with

    Returns:
        out: list containig  file_name, predictnr, GA, landmark error wrt A_t and A_b
        mov: moved image
        T: affine transformation matrix
    """

    out=[]
    X_vol=datagenerators.load_volfile(img)[np.newaxis, ..., np.newaxis]
    GA=np.load(img.split('.nii')[0]+'_annotation.npz')['GA']

    X_t = np.load(img.split('.nii')[0]+'_annotation.npz')['coor_t']
    X_b = np.load(img.split('.nii')[0]+'_annotation.npz')['coor_b']
    predictnr = get_predict_nr(img)
    [mov,T] = model.predict([X_vol])

    landmark_error = calculate_landmark_error(T, A_t, A_b, X_t, X_b)
    out.append(img)
    out.append(predictnr)
    out.append(GA)
    out.append(landmark_error)
    
    return out, mov, T

def calculate_landmark_error(warp_global,A_t, A_b, X_t, X_b, size=32):
    """
    Function to calculate the landmark error.

    Args:
        warp_global: affine transformation matrix
        A_t: coordinate of top landmark in atlas
        A_b: coordinate of bottom landmark in atlas
        X_t: annotation of the top landmark in the image
        X_b: annotation of the bottom landmark in the image

    Returns:
        landmark error given annotations X_t, X_b wrt A_t, B_t using warp_global.
    """
    y_pred=np.concatenate((np.reshape(warp_global,[3,4]),np.zeros([1,4])),axis=0)+np.identity(4)
    x_t=np.array([size,size,size])+np.matmul(y_pred,np.concatenate((A_t-np.array([size,size,size]),np.array([1,])),axis=0))[:-1]
    x_b=np.array([size,size,size])+np.matmul(y_pred,np.concatenate((A_b-np.array([size,size,size]),np.array([1,])),axis=0))[:-1]
    landmark_error = (np.linalg.norm(X_t-x_t)+np.linalg.norm(X_b-x_b))/2

    return landmark_error

def save_png(im_result,labels,save_dir,slice_number,size_x,size_y,axis=2,rows=5,columns=10):
    """
    Simple script to view a list of 3D image in a rows x colums raster.
    Each images is viewed in the same slice. The file is saved as a PNG file
    
    Args:
        im_result: list of images/segmentations
        labels: title/label/class/filename of this image
        save_dir: directory to save the .png image
        slice_number: The slice to show
        size_x: dimension of image in x axis
        size_y: dimension of image in y axis
        axis: axis to take slices in
        rows: number of rows of images
        columns: number of columns of images  

    Returns:
        saved png
    """
    
    fig=plt.figure()    

    for num in range(1,rows*columns+1):
        fig.add_subplot(rows,columns,num)
        if axis == 2:
            plt.imshow(im_result[num-1][:,:,slice_number], vmin=0,  vmax=1)
        elif axis == 1:
            plt.imshow(im_result[num-1][:,slice_number,:], vmin=0, vmax=1)
        else:
            plt.imshow(im_result[num-1][slice_number,:,:], vmin=0, vmax=1)        
        
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel(labels[num-1],fontsize = 'xx-small')    

    fig.savefig(save_dir+'/visual_result_axis_' + str(axis),dpi=300)

