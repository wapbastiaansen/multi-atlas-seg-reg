from multi_affine.losses import *
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

def test_landmark():
    A_t = [17,0,0]
    A_b = [47,0,0]
    landmark_loss = Landmark_loss(A_t = A_t, A_b = A_b)
    
    y_pred = np.zeros((1,12))
    y_true = np.zeros((1,12))
    y_true[0,0:3] = [47,0,0]
    y_true[0,3:6] = [17,0,0]
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

    loss = landmark_loss.loss(y_true, y_pred)
    error = loss.numpy()
    assert np.sum(error - 30) < 1e-5

def test_scaling_loss():
    y_true = np.zeros((1,12))
    y_pred = np.zeros((1,12))

    y_pred[0,0] = 0.2
    y_pred[0,5] = 0.2
    y_pred[0,10] = 0.2

    y_pred = tf.convert_to_tensor(y_pred, dtype = tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype = tf.float32)

    scaling_loss=Scaling_loss()
    loss = scaling_loss.loss(y_true, y_pred)
    error=loss.numpy()
    assert np.sum(error - 0.0997) < 1e-4

def test_masked_NCC():
    mask = np.ones((1,64,64,64,1),dtype=np.float32)
    multi_atlas = np.load('test_affine/test_data/atlas_noise.npz')['atlas_vol']
    y_true = np.zeros((1,64,64,64,1))
    y_pred = np.ones((1,64,64,64,1))
    multi_atlas[0,:,:,:,:] = y_pred
    
    y_true[0,0,0,0,0] = 0

    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    sim_loss = masked_NCC(M=1, mask=mask, multi_atlas=multi_atlas)

    loss = sim_loss.loss(y_true,y_pred)
    error = loss.numpy()
    assert np.sum(error - -0.33) < 0
    
    y_true = np.zeros((1,64,64,64,1))
    y_true[0,0,0,0,0] = 0 
    y_true[0,1,0,0,0] = 1
    y_true[0,2,0,0,0] = 4
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

    sim_loss = masked_NCC(M=3, mask=mask, multi_atlas=multi_atlas)
    loss = sim_loss.loss(y_true,y_pred)
    assert error < -0.33
    
    multi_atlas = np.ones((1,64,64,64,1))

    sim_loss = masked_NCC(M=1, mask=mask, multi_atlas=multi_atlas)
    loss = sim_loss.loss(y_true, y_pred)
    assert np.sum(error -- 0.33) < 0

    y_true = np.zeros((1,64,64,64,1))
    y_true[0,0,0,0,0] = 0
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    sim_loss = masked_NCC(M=47, mask=mask, multi_atlas=multi_atlas, mode = 'give1')
    loss = sim_loss.loss(y_true, y_pred)
    assert np.sum(error - -0.33) < 0 
