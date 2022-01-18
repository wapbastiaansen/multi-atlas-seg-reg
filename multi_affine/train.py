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
from multi_affine import datagenerators
from multi_affine import networks
from multi_affine import losses
from multi_affine.utils import load_multi_atlas, summary_experiment


def train(atlas_dir,
          data_dir,
          model_dir,
          reg_param,
          stage,
          num_iterations,
          atlas_list,
          M,
          mask_file,
          enc_global,
          load_model_file,
          learning_rate,
          data_aug = [0,0],
          num_fcl = 4,
          nhidden = 1000):
    """
    Function to launch trainig of affine network.

    Args:
        atlas_dir: folder with npz files of all atlasses used.
        data_dir: folder with npz files for each subject.
        model_dir: the model directory to save to
        reg_param: trade off parameter
        stage: either 1 (supervised, landmark) or 2 (unsupervised)
        num_iterations: number of training iterations
        atlas_list: list with id of atlases used
        M: number of closed (in age) atlasses used for optimziation
        mask_file: mask to be used
        enc_global: list of the number of filters in each layer of the encoder
        load_model_file: weights to load in stage 2, None to train from beginning
        learning_rate: the learning rate used
        data_aug: parameter indicating data augmentation used. [0,0]:none, [1,0]: only flips, [0,1]: only rot90, [1,1]: flips or rot90
        num_fcl: number of fcl
        nhidden: number of neurons per fcl


    Returns:
        folder with log.csv (log of the loss), summary_experiment.xt (txt file with all parameters and data information) .h5 weight files
    """
    
    # load atlas and metadata from provided files.
    atlasses, segs, Age, atlas_files, A_t, A_b  = load_multi_atlas(atlas_dir, atlas_list, True, True, False)
    vol_size = atlasses.shape[1:-1] 
    
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        
    # save summary of performed experiment    
    parameters = [atlas_files, data_dir, model_dir, reg_param, num_iterations, atlas_list, M, mask_file, enc_global, load_model_file, learning_rate, data_aug, num_fcl, nhidden]
    names = ['atlas_files','data_dir','model_dir', 'reg_param', 'num_iterations', 'atlas_list', 'M', 'mask_file', 'enc_global', 'load_model_file', 'learning_rate','data_aug', 'num_fcl', 'nhidden']
    summary_experiment(model_dir, parameters, names)

    # load data
    train_vol_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    val_vol_names = glob.glob(os.path.join(data_dir.replace('train', 'val'), '*.nii.gz'))

    
    random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"
    # the number of steps per epoch depends on wether or not we use data augmentation.
    if np.sum(data_aug) > 0:
        steps_per_epoch = 10000
    else:
        steps_per_epoch = len(train_vol_names)

    batch_size = 1

    loss_weights = [1, reg_param]

    # load the losses
    if stage == '1':
       mask = np.load(mask_file)['mask'][np.newaxis,...,np.newaxis]
       data_loss = losses.masked_NCC(M = M, mask = mask,multi_atlas = atlasses).loss
       warp_loss = losses.Landmark_loss(A_t = A_t,A_b = A_b).loss
       loss_terms = [data_loss, warp_loss]
    if stage == '2':
        mask = np.load(mask_file)['mask'][np.newaxis,...,np.newaxis]
        data_loss = losses.masked_NCC(M = M, mask = mask, multi_atlas = atlasses).loss
        warp_loss = losses.Scaling_loss().loss
        loss_terms = [data_loss, warp_loss]
    
    # load model
    model = networks.network_multi_affine(vol_size, enc_global, num_fcl, nhidden)
    
    # load initial weights
    if load_model_file is not None:
        print('loading', load_model_file)
        model.load_weights(load_model_file)
    
    # save first iteration
    model.save(os.path.join(model_dir, '0.h5'))

    # load data generator
    train_image_gen = datagenerators.image_generator(train_vol_names, stage, M, Age, data_aug, atlas_files, batch_size = batch_size)
    val_image_gen = datagenerators.image_generator(val_vol_names, stage, M, Age, [0,0], atlas_files, batch_size = batch_size)
    Gen = datagenerators.datagenerator_affine(train_image_gen)
    Gen_val = datagenerators.datagenerator_affine(val_image_gen)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:002d}.h5')

    # fit generator
    if stage == '1':
        save_callback = ModelCheckpoint(save_file_name, save_weights_only = True, period = 10)
    if stage == '2':
        save_callback = ModelCheckpoint(save_file_name, save_weights_only = True, period = 1)

    csv_logger = CSVLogger(model_dir+'/log.csv', append = True, separator = ',')
    

    model.compile(optimizer = Adam(lr = learning_rate), 
                     loss = loss_terms,
                     loss_weights = loss_weights)
                
    # fit
    model.fit_generator(Gen, 
                        initial_epoch = 0,
                        epochs = num_iterations,
                        callbacks = [save_callback,csv_logger],
                        validation_data = Gen_val, 
                        validation_steps = len(val_vol_names),
                        steps_per_epoch = steps_per_epoch,
                        verbose = 1)
