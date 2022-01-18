import numpy as np
import nibabel as nib
import random

def datagenerator_affine(gen,test = False):
    """ 
    Datagenerator for the affine network (both stages)

    Args:
        gen: an image generator, loading the image, indicator, loss placeholders and landmarks
        test: True in case we test the function

    Returns:
        generator to train the affine network using Keras
    """

    while True:
        X = next(gen)
        yield ([X[0]], [X[1], X[2]])
        if test == True:
            return [X[0], [X[1],X[2]]]
            
        
def image_generator(vol_names, stage, M, age_atlas, data_aug, atlas_files, mode = 'giveall', test = False,  batch_size=1, np_var='vol_data'):
    """
    Generator that loads an image along with placeholders for the loss

    Args:
        vol_names: list with all files for the generator
        stage: string, '1' for stage 1, '2' for stage 2
        age_atlas: ordered list with all GA of the atlas images
        M: number of atlases used to calculate the loss
        data_aug: type of data augmentation used: [0,0] no augmentation, [1,0] only flips, [0,1] only rot90, [1,1] flip or rot is applied
        atlas_files: ordered list of names of atlas files
        mode: used training strategy: 'giveall': means give all M atlases to loss, give1: means give at random 1 of M atlases to loss
        test: default (False) set to True to test the function

    Returns:
        image generator to train/validate the affine network using Keras
    """

    while True:
        idxes = np.random.randint(len(vol_names),size=batch_size)
        
        for idx in idxes:
            X_orig = load_volfile(vol_names[idx], np_var=np_var)
            
            gt_file = vol_names[idx].split('.nii')[0]+'_annotation.npz'
            
            X, Y = loader_img_anno(X_orig, gt_file, stage, data_aug)
        
            X = X[np.newaxis, ..., np.newaxis]
            
            XX = loader_age(gt_file, M, age_atlas, X, atlas_files, mode)

            return_vals = [X,XX,Y]
            
        yield tuple(return_vals)
        if test == True:
            return return_vals
        

def loader_age(gt_file, M, age_atlas, X, atlas_files, mode = 'giveall'):
    """
    Function that loads the placeholder in the loss for image simmialirty, 
    contains (by design) the indicator list that tells which atlases are eligible.

    Args:
        gt_file: npz file containing the meta-data per image. Here we load the GA.
        M: number of atlases used to calculate the loss.
        age_atlas: ordered list of GA of atlas files
        X: The image for which we will load the placeholder.
        atlas_files: ordered list of names atlas files
        mode: used training strategy: 'giveall': means give all M atlases to loss, give1: means give at random 1 of M atlases to loss

    Returns:
        XX: the placeholder for the loss, same shape as X, containing zeros and on
        (0,indi,0,0,0) the indicator.
    """

    age = int(load_volfile(gt_file,np_var='GA'))
    indi = indicator(age, M, age_atlas, atlas_files)
    XX = np.zeros(X.shape)

    if mode == 'give1':
        idx = select_index_atlas(indi)
        XX[0,0,0,0,0] = idx
    else:
        idx = give_index_atlas(indi)
        XX[0,0:M,0,0,0] = idx
    
    return XX

def loader_img_anno(X_orig, gt_file, stage,  data_aug, test=False,test_flips=False):
    """
    Function that loads the image, applies data augmenation  and load the annotations of the landmarks used in the datagenerator.
    Args:
        X_orig: Original image loaded, for this image we will load accompanying information.
        gt_file: corresponding npz file with meta-data: age and landmarks
        data_aug: type of data augmentation used: [0,0] no augmentation, [1,0] only flips, [0,1] only rot90, [1,1] flip or rot is applied
        test: default (False) set to True for testing.
        test_flips: defaul (False), set to True to test.

    Returns:
        X: image used for trainig
        Y: (1,12) vector containing the landmarks for stage 1, containing zeros for stage 2.
    """
    if stage == '1':
        Y_t1 = load_volfile(gt_file, np_var='coor_t')
        Y_b1 = load_volfile(gt_file, np_var='coor_b')
    else:
        Y_t1 = []
        Y_b1 = []

    if np.sum(data_aug) == 2:
        if test == True:
            flips = test_flips
        else:
            flips = random.choice([True, False])
        if flips == True:
            if test == True:
                X, Y_t, Y_b = apply_flips(X_orig, Y_t1, Y_b1, size = 32, print_flip = False, test = True, test_lr = True)
            else:
                X, Y_t, Y_b = apply_flips(X_orig, Y_t1, Y_b1)
        else:
            if test == True:
                X, Y_t, Y_b = apply_rot90(X_orig, Y_t1, Y_b1, size = 32, print_rot90 = False, test = True, r_test = 0)
            else:
                X, Y_t, Y_b = apply_rot90(X_orig, Y_t1, Y_b1)
    elif data_aug[0] == 1:
        if test == True:
            X,Y_t,Y_b = apply_flips(X_orig, Y_t1, Y_b1, size = 32, print_flip = False, test = True, test_lr = True)
        else:
            X,Y_t,Y_b = apply_flips(X_orig, Y_t1, Y_b1)
    elif data_aug[1] == 1:
        if test == True:
            X,Y_t,Y_b = apply_rot90(X_orig, Y_t1, Y_b1, size = 32, print_rot90 = False, test = True, r_test = 0)
        else:
            X,Y_t,Y_b = apply_rot90(X_orig, Y_t1, Y_b1)    
    else:
        X = X_orig
        Y_t = Y_t1
        Y_b = Y_b1
        
    Y = np.zeros([1,12])
    if stage == '1':
        Y[0,0:3] = Y_t
        Y[0,3:6] = Y_b
    
    return X, Y
    
def apply_flips(img, T, B, size = 32, print_flip = False,test = False, test_lr = False,test_ud = False,test_fb = False):   
    """
    Function that applies flips to img.

    Args:
        img: image that will be augmented
        T: top landmark that will be augmented
        B: bottum landmark that will be augmented
        size: 0.5*shape of img
        print_flips: if set to True the applied flip will be printed
        test: set to True to test the code
        test_lr: in case of test, control if lr flip is applied
        test_ud: in case of test, control if ud flip is applied
        test_fb: in case of test, control if fb flip is applied

    Returns:
        img_aug: image augmented
        T_aug: top landmark augmented
        B_aug: bottum landmark augmented
    """
    if test  == True:
        fliplr = test_lr
    else:
        fliplr = random.choice([True,False])

    if fliplr == True:
        img = np.fliplr(img)
        if T != []:
            T[1] = 2*size-T[1]
            B[1] = 2*size-B[1]

    if test == True:
        flipud = test_ud
    else:        
        flipud = random.choice([True, False])
    
    if flipud == True:
        img = np.flipud(img)
        if T != []:
            T[0] = 2*size-T[0]
            B[0] = 2*size-B[0]

    if test == True:
        flipfb = test_fb
    else:
        flipfb = random.choice([True,False])
    
    if flipfb == True:
        img = np.flip(img,2)
        if T != []:
            T[2] = 2*size-T[2]
            B[2] = 2*size-B[2]
        
    if print_flip == True:
        print('fliplr: '+ str(fliplr))
        print('flipud: '+ str(flipud))
        print('flipfb: '+ str(flipfb))
        
    return img, T, B
 
def apply_rot90(img, T, B, size = 32, print_rot90 = False, test = False, r_test = []):
    """
    Function that applies 90 degree rotations to img.

    Args:
        img: image that will be augmented
        T: top landmark that will be augmented
        B: bottum landmark that will be augmented
        size: 0.5*shape of img
        print_rot90: if set to True the applied rotation will be printed
        test: set to True to test the code
        r_test: in case of test, control which rotation is applied
 
    Returns:
        img_aug: image augmented
        T_aug: top landmark augmented
        B_aug: bottum landmark augmented
    """
    #list of all rotations we consider.
    rots = [[90,[0,0,1],0,1,1],
           [-90,[0,0,1],1,0,1],
           [90,[1,0,0],1,2,1],
           [-90,[1,0,0],2,1,1],
           [90,[0,1,0],2,0,1],
           [-90,[0,1,0],0,2,1],
           [180,[0,0,1],0,1,2],
           [180,[1,0,0],1,2,2],
           [180,[0,1,0],0,2,2],
           [0, [0,0,0],0,0,0]]

    if test == True:
        r = r_test
    else:
        r = random.randint(0,len(rots)-1)
    if r != 9:
        img = np.rot90(img,rots[r][4],axes=(rots[r][2],rots[r][3]))
        if T != []:
            theta = np.deg2rad(rots[r][0])
            v = rots[r][1]
            rotation_matrix = np.array([[v[0]*v[0]*(1-np.cos(theta))+np.cos(theta), v[1]*v[0]*(1-np.cos(theta))-v[2]*np.sin(theta), v[2]*v[0]*(1-np.cos(theta))+v[1]*np.sin(theta),0],
                                        [v[0]*v[1]*(1-np.cos(theta))+v[2]*np.sin(theta), v[1]*v[1]*(1-np.cos(theta))+np.cos(theta), v[1]*v[2]*(1-np.cos(theta))-v[0]*np.sin(theta),0 ],
                                        [v[0]*v[2]*(1-np.cos(theta))-v[1]*np.sin(theta),v[1]*v[2]*(1-np.cos(theta))+v[0]*np.sin(theta),v[2]*v[2]*(1-np.cos(theta))+np.cos(theta),0],
                                        [0,0,0,1]])
    
            
            T = np.array([size,size,size])+np.matmul(rotation_matrix,np.concatenate((T-np.array([size,size,size]),np.array([1,])),axis=0))[:-1]
            B = np.array([size,size,size])+np.matmul(rotation_matrix,np.concatenate((B-np.array([size,size,size]),np.array([1,])),axis=0))[:-1]
    
    if print_rot90 == True:
        print('rot90: '+'degree: '+str(rots[r][0])+'axis: '+str(rots[r][2])+' '+str(rots[r][3]))
        
    return img, T, B
       
def load_volfile(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names in np_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):

        X = nib.load(datafile).get_fdata()
        
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X

def indicator(age, num_atlas, age_atlas, atlas_files, mode='nn'):
    """
    Gives back indicator list to indicate which atlases will be used to calculate the loss. The atlasses picked
    are closest in GA (in NN sense), in case of equal distance the ones first seen in the list are chosen.
    
    Args:
        age: (estimated) gestational age of the subject
        num_atlas: number of atlases considered in the loss
        age_atlas: sorted list with GA of all atlas files
        atlas_files: sorted list with names of all atlas files (by GA)
        mode: so far only 'nn' is implemented: linear interpolation idea could be future suggestion
    
    Returns:
        indicator: list of length N containing M ones.
    """
    
    assert(num_atlas<len(age_atlas))

    diff = []
    for i in range(len(age_atlas)):
        
        diff.append(abs(age-age_atlas[i][0]))
    
    if mode == 'nn':    
        xx = np.argsort(diff)
        print(xx) 
        indicator = np.zeros([len(age_atlas),])
        i = 0
        j = 0
        atlas_set = []
        while i<num_atlas:
            if atlas_files[xx[j]].split('atlas_')[1][:5] not in atlas_set:
                indicator[xx[j]] = 1
                atlas_set.append(atlas_files[xx[j]].split('atlas_')[1][:5])
                i+=1
                print(atlas_set)
            j+=1
                   
    return indicator


def select_index_atlas(indicator):
    """
    function to select available atlas from indicator.

    Args:
        indicator: list with ones and zeros, indicating with a one that the atlas is available

    Returns:
        index of selected atlas
    """

    idx = []
    for i in range(len(indicator)): 
        if indicator[i] == 1:
            idx.append(i)

    return idx[np.random.randint(len(idx))]

def give_index_atlas(indicator):
    """
    function to give idx for available atlases.

    Args:
        indicator: list with ones and zeros indicating with a one that the atlas is available

    Returns:
        indexes of available atlases

    """

    idx = []
    for i in range(len(indicator)):
        if indicator[i] == 1:
            idx.append(i)

    return idx
