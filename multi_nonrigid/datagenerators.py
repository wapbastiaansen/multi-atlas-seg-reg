import numpy as np
from multi_affine.datagenerators import indicator, load_volfile, select_index_atlas

def datagenerator_nonrigid(gen, diffeomorphic=False, atlas_shape=[64,64,64], batch_size=1, test=False):
    """ 
    function to generate data for training.

    Args:
        gen: image generator to load image and atlas
        diffeomorphic: if true placeholder for inv_warp is added

    Returns:
        input [image, atlas] and output network [atlas, zeros] where the zeros are a placeholder for the deformation field.
    """
    
    zeros = np.zeros((batch_size, *atlas_shape, len(atlas_shape)))
    
    if diffeomorphic == False:
        while True:
            X = next(gen)
            yield ([X[0], X[1]], [X[1], zeros]) 
            if test == True:
                return [[X[0], X[1]], [X[1], zeros]]
    else:
        while True:
            X = next(gen)
            yield ([X[0], X[1]], [X[1], zeros, zeros])
            
def image_generator(vol_names, M, atlasses, age_atlas, atlas_files, batch_size=1, np_var='vol_data', test=False):
    """
    function to load image and atlas for training.

    Args:
        vol_names: list with names of the images in the dataset
        M: number of atlases used for optimization
        atlasses: npy array with atlasses
        age_atlas: list with age of atlases
        atlas_list: list of names of avaliable atlases

    Return:
        X: input image
        atlas: select atlases to be used as input
        Here out of the M eligible atlases one is chosen as input for the network.
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        for idx in idxes:
            X = load_volfile(vol_names[idx], np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
            # for implementation of data augmentation: this should be put here

            gt_file=vol_names[idx].split('_moved_affine')[0]+'_annotation.npz'
            
            age=int(load_volfile(gt_file,np_var='GA'))
            
            indi=indicator(age, M, age_atlas, atlas_files)
            
            #randomly select which of the M available atlases from indi is used for training in this epoch
            idxes_select=select_index_atlas(indi)
            
            atlas=atlasses[idxes_select,:,:,:,:][np.newaxis,...]
        
        return_vals = [X,atlas]

        yield tuple(return_vals)
        if test == True:
            return return_vals
        

