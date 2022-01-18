# python imports
import os
import glob
import random
import numpy as np


# third-party imports
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

# project imports
from multi_nonrigid import datagenerators
from multi_nonrigid import networks
from multi_nonrigid import losses
import multi_affine.losses as losses_affine
from multi_affine.utils import load_multi_atlas, select_on_GA, summary_experiment


def train(atlas_dir,
          data_dir,
          model_dir,
          reg_param,
          num_iterations,
          mask_file,
          atlas_list,
          load_model_file,
          M,
          enc,
          dec,
          diffeomorphic=False,
          int_steps = 7,
          week_nr = 'all',
          test = False,
          learning_rate=0.0001):
    """
    Function to launch trainig of affine network.

    Args:
        atlas_dir: folder with npz files of all atlasses used.
        data_dir: folder with npz files for each subject.
        model_dir: the model directory to save to
        reg_param: trade off parameter
        num_iterations: number of training iterations
        load_model_file: optional h5 model file to initialize with
        atlas_list: list with id of atlases ued
        M: number of close (in age) atlases used for training (selected from randomly)
        mask_file: mask to be used for similairity loss
        enc: list with number of filters in enc
        dec: list with number of filters in dec
        diffeomorphic: if true the diffeomorphic model is trained
        int_steps: for diffeomorphic model number of integration steps used
        week_nr: 'all' if data 8-12 weeks is used, otherwise indicate on which week we train

    Returns:
        folder with locg.csv (log of the loss), summary_experiment.txt (tst file with all parameters and data information) .h5 weight files

    """

    # load atlas and metadata from provided files.
    atlasses, segs, Age, atlas_files, A_t, A_b  = load_multi_atlas(atlas_dir, atlas_list, True, True, False)
    vol_size = atlasses.shape[1:-1]
 
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
 
    # save summary of performed experiment
    parameters = [atlas_files, data_dir, model_dir, reg_param, num_iterations, atlas_list, M, mask_file, enc, dec, load_model_file, learning_rate, week_nr, diffeomorphic, int_steps]
    names = ['atlas_files','data_dir','model_dir', 'reg_param', 'num_iterations', 'atlas_list', 'M', 'mask_file', 'enc', 'dec',  'load_model_file', 'learning_rate', 'week_nr', 'diffeomorphic','int_steps']
    summary_experiment(model_dir, parameters, names)

    
    # prepare data files
    train_vol_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    val_vol_names = glob.glob(os.path.join(data_dir.replace('train', 'val'), '*.nii.gz'))
    if week_nr!='all':
        train_vol_names = select_on_GA(train_vol_names, week_nr)
        val_vol_names = select_on_GA(val_vol_names, week_nr)

    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"
    
    steps_per_epoch=len(train_vol_names)

    #definition of the loss
    if diffeomorphic == True:
        loss_weights = [1,reg_param,0]
    else:    
        loss_weights=[1,reg_param]

    mask=np.load(mask_file)['mask'][np.newaxis,...,np.newaxis]
    mask_sum=np.sum(mask)

    #M=1 here because we always select on atlas to give for training
    data_loss = losses_affine.masked_NCC(mask=mask, mode='nonrigid').loss
    diff_loss=losses.grad_3D(mask_sum=mask_sum).loss
    zero_loss=losses.zeros_loss().loss

    if diffeomorphic == True:
        loss_terms = [data_loss, diff_loss, zero_loss]    
    else:
        loss_terms=[data_loss, diff_loss]
        
    # load model
    nf_enc = enc
    nf_dec = dec

    
    model=networks.network_nonrigid(vol_size, nf_enc, nf_dec, diffeomorphic=diffeomorphic, int_steps=int_steps)
    
    # load initial weights
    if load_model_file is not None:
        print('loading', load_model_file)
        model.load_weights(load_model_file)
    
    #save first iteration
    model.save(os.path.join(model_dir, '0.h5'))

    # data generator


    train_image_gen=datagenerators.image_generator(train_vol_names, M, atlasses, Age, atlas_files)
    val_image_gen = datagenerators.image_generator(val_vol_names, M, atlasses, Age, atlas_files)

    gen = datagenerators.datagenerator_nonrigid(train_image_gen, diffeomorphic)
    gen_val = datagenerators.datagenerator_nonrigid(val_image_gen, diffeomorphic)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:002d}.h5')

    # fit generator
    save_callback = ModelCheckpoint(save_file_name,save_weights_only=True,period=10)
    csv_logger=CSVLogger(model_dir+'/log.csv', append=True, separator=',')
    


    model.compile(optimizer=Adam(lr=learning_rate), 
                     loss=loss_terms,
                     loss_weights=loss_weights)
                
    # fit
    model.fit_generator(gen, 
                        initial_epoch=0,
                        epochs=num_iterations,
                        callbacks=[save_callback,csv_logger],
                        validation_data=gen_val,
                        validation_steps=len(val_vol_names),
                        steps_per_epoch=steps_per_epoch,
                        verbose=1) 
