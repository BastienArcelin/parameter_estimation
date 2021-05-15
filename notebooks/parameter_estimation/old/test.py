import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
#from sklearn import preprocessing
from importlib import reload
import matplotlib as mpl
import scipy
import os

sys.path.insert(0,'../../scripts/tools_for_VAE/')
import tools_for_VAE.model
#from tools_for_VAE import utils, vae_functions, generator, model, boxplot, plot

######## Parameters
nb_of_bands = 6
batch_size = 256

input_shape = (59, 59, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 2
filters = [32,64,128,256]
kernels = [3,3,3,3]

conv_activation = None
dense_activation = None

bands = [0,1,2,3,4,5]

root_dir_v1 = '/pbs/home/b/barcelin/sps_link/data/dc2_test/training_24.5_v2/' # mag cut at 24.5 in r band
#root_dir_v2 = '/pbs/home/b/barcelin/sps_link/data/dc2_test/test_mag_26.5/' # mag cut at 26.5 in r band

images_noiseless = np.load(root_dir_v1+'img_noiseless_sample_1.npy', mmap_mode = 'c') # artificial augmentation of pixel intensity. Residual from tests, needs to be suppressed
images_blend = np.load(root_dir_v1+'img_cropped_sample_1.npy', mmap_mode = 'c')
psf = np.load(root_dir_v1+'psf_cropped_sample_1.npy', mmap_mode = 'c')
data = pd.read_csv(root_dir_v1+'img_noiseless_data_1.csv')

images_noiseless = images_noiseless[:512]
images_blend = images_blend[:512]
psf = psf[:512]

ell_1 = np.array(data['e1'])
ell_2 = np.array(data['e2'])
shear_1 = np.array(data['shear_1'])
shear_2 = np.array(data['shear_2'])
convergence = np.array(data['convergence'])


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Multiply, Reshape, Flatten, BatchNormalization, Conv2D, Conv3D, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda,  concatenate
import tensorflow_probability as tfp


tfd = tfp.distributions
encoded_size = 32
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)


def create_model_wo_ls_peak_pooling_vae(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = MaxPool2D()(h)
        #h_2 = MaxPool2D()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        #h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    #h = Flatten()(h)
    #h_2 = Flatten()(h_2)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    #h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(1/x, tf.complex64), name=None))(h_2)

    #h = tf.keras.layers.Multiply()([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)])
    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(256)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(32),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(32,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))#,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.001)
    
    
    
    h = PReLU()(h)
    h = Dense(256)(h)
    h = PReLU()(h)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    w = int(np.ceil(input_shape[0]/2**(len(filters))))
    h = Dense(w*w*filters[-1], activation=dense_activation)(tf.cast(h,tf.float32))
    h = PReLU()(h)
    h = Reshape((w,w,filters[-1]))(h)
    for i in range(len(filters)-1,-1,-1):
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h = tf.keras.layers.UpSampling2D()(h)
        h = Conv2DTranspose(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
    h = Conv2D(input_shape[-1], (3,3), activation='sigmoid', padding='same')(h)
    #h = PReLU()(h)
    cropping = int(h.get_shape()[1]-input_shape[0])
    if cropping>0:
        print('in cropping')
        if cropping % 2 == 0:
            h = Cropping2D(cropping/2)(h)
        else:
            h = Cropping2D(((cropping//2,cropping//2+1),(cropping//2,cropping//2+1)))(h)


    model = Model([input_layer_1, input_layer_2],h)

    return model



path_output = '/pbs/home/b/barcelin/sps_link/TFP/weights/test_dc2/v19/mse/'
net = create_model_wo_ls_peak_pooling_vae(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)

latest = tf.train.latest_checkpoint(path_output)
net.load_weights(latest)

#### Loss definition
model_loss = 'crossentropy'
if model_loss == 'mse':
    vae_loss = tf.keras.losses.MeanSquaredError()
elif model_loss == 'crossentropy':
    def vae_loss(x, x_decoded_mean):
        xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
        #x_res = K.mean(K.sum(K.binary_crossentropy(x, x-x_decoded_mean), axis=[1,2,3]))
        return xent_loss


# Custom metrics
def kl_metric(y_true, y_pred):
    return K.sum(net.losses)
mse = tf.keras.losses.MeanSquaredError()

images_blend = np.tanh(np.arcsinh(images_blend))
images_noiseless = np.tanh(np.arcsinh(images_noiseless))
psf = np.tanh(np.arcsinh(psf))

for i in range (len(net.layers[44:])):
                net.layers[44+i].trainable = False
        
net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
              loss=vae_loss,
              metrics = ['mse', 'acc',kl_metric],
              #experimental_run_tf_function=False,
              )

######## Train the network
## With dataset (faster than directly from generator)
hist = net.fit((images_blend, psf), images_noiseless, epochs=20,#training_ds
                    steps_per_epoch=10,
                    verbose=1,
                    shuffle=True,
                    validation_data=((images_blend, psf), images_noiseless),#validation_ds
                    validation_steps=3,)
                    #callbacks=[tb_callback])

