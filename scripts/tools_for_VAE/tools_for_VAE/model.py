# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
import galsim
import random
import cmath as cm
import math
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LocallyConnected2D, Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

import tensorflow as tf
tfd = tfp.distributions

sys.path.insert(0,'../../scripts/tools_for_VAE/')
from tools_for_VAE import ktied_distribution

def create_model_old(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    tfd = tfp.distributions
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
                            reinterpreted_batch_ndims=1)

    input_layer = Input(shape=(input_shape)) 

    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(
            latent_dim,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(h)

    # Decoding part
    h = Flatten()(h)
    h = tf.keras.layers.Dense(64, activation=None)(h) # 512
    h = tf.keras.layers.PReLU()(h)

    # Multivariate gaussian
    h = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),activation=None)(h) #'relu'
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model(input_layer,h)

    return model

def create_model_wo_ls(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer = Input(shape=(input_shape)) 
    
    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)

    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model(input_layer,h)

    return model



# Probabilistic models

import tensorflow.compat.v1 as tf1
from tensorflow_probability.python.layers import util as tfp_layers_util
# Weights initialization for posteriors
def get_posterior_fn():
  return tfp_layers_util.default_mean_field_normal_fn(
      loc_initializer=tf1.initializers.he_normal(), 
      untransformed_scale_initializer=tf1.initializers.random_normal(
          mean=-15, stddev=0.1)#mean=-9, stddev=0.1)
      )
# kernel divergence weight in loss
kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p) / (10000))

def create_model_full_prob_rt(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):

    input_layer = Input(shape=(input_shape)) 
    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]), 
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]), 
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', 
                                            strides=(2,2))(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = tfp.layers.DenseReparameterization(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=dense_activation)(h)

    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model(input_layer,h)
    
    return model

def create_model_full_prob_flipout(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    
    input_layer = Input(shape=(input_shape)) 
    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(), 
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)


    model = Model(input_layer,h)
    
    return model


# Model with coordinate of target galaxy
def create_model_wo_ls_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    tfd = tfp.distributions
    # Normal ditribution prior
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(final_dim), scale=1),
                        reinterpreted_batch_ndims=1)
    # Create a mixture of two Gaussians for prior
    mix = 0.3
    bimix_gauss = tfd.Mixture(
    cat=tfd.Categorical(probs=[mix, 1.-mix]),
    components=[
    tfd.Normal(loc=-0., scale=0.03),
    tfd.Normal(loc=+0., scale=0.4),
    ])
    ### Test adding special activation function:
    # def mapping_to_target_range( x, target_min=-1, target_max=1 ) :
    #     x /=np.max(x)
    #     x -= np.min(x)
    #     scale = ( target_max-target_min )/2.
    # return  x02 * scale + target_min

    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)
    
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    #h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h_2)

        h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
        #h = tf.keras.layers.SeparableConv2D(filters[i], (1,1), activation=None, padding='valid',  data_format='channels_last')(h)
        #h = tf.keras.layers.multiply([h,h_2]) #

    h = Flatten()(h)
    h = PReLU()(h)
    
    #h = Dense(64)(h)
    #h = PReLU()(h)
    #h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h) #Dense(2)(h)#
    #activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.2))(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model

# Model with coordinate of target galaxy
def create_model_wo_ls_peak_3(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model

# Model with coordinate of target galaxy
def create_model_wo_ls_peak_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    tfd = tfp.distributions

    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)
    
    h = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h)
    h_2 = tf.keras.layers.Lambda(lambda x: tf.signal.fft2d(tf.cast(x, tf.complex64), name=None))(h_2)

    h = tf.keras.layers.concatenate([tf.cast(h,tf.float64), tf.cast(h_2,tf.float64)], axis =-1)
        #h = tf.keras.layers.SeparableConv2D(filters[i], (1,1), activation=None, padding='valid',  data_format='channels_last')(h)
        #h = tf.keras.layers.multiply([h,h_2]) #

    h = Flatten()(h)
    h = PReLU()(h)
    
    h = Dense(64)(h)
    h = PReLU()(h)
    h = tf.keras.layers.Lambda(lambda x: tf.signal.ifft(tf.cast(x, tf.complex64), name=None))(h)
    
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h) #Dense(2)(h)#
    #activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.2))(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model

# Model with coordinate of target galaxy
def create_model_wo_ls_inception(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):

    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2

    for i in range(len(filters)):
        #h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        #h_2 = PReLU()(h_2)
        #h_2_1 = Conv2D(filters[i], (1,1), activation=None, padding='same')(h_2)
        #h_2_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2_1)
        #h_2_2 = Conv2D(filters[i], (1,1), activation=None, padding='same')(h_2)
        #h_2_2 = Conv2D(filters[i], (5,5), activation=None, padding='same')(h_2_2)
        #h_2_3 = Conv2D(filters[i], (1,1), activation=None, padding='same')(h_2)
        #h_2 =tf.keras.layers.concatenate([h_2_1, h_2_2,h_2_3], axis =-1)
        #h_2 = PReLU()(h_2)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)

        h_1 = Conv2D(filters[i], (1,1), activation=None, padding='same')(h)
        h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_1)
        h_1_1 = Conv2D(filters[i], (1,1), activation=None, padding='same')(h)
        h_1_1 = Conv2D(filters[i], (5,5), activation=None, padding='same')(h_1_1)
        h_1_2 = Conv2D(filters[i], (1,1), activation=None, padding='same')(h)
        h_11 =tf.keras.layers.concatenate([h_1, h_1_1,h_1_2], axis =-1)
        #h = h_11+h
        h = PReLU()(h_11)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
        #h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        #h_2 = PReLU()(h_2)

    #h = tf.keras.layers.concatenate([h, h_2], axis =-1)

    h = Flatten()(h)
    h = PReLU()(h)

    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)#,
        #activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.2))(h)
    model = Model([input_layer_1, input_layer_2],h)

    return model


def create_model_wo_ls_resnet(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h_1 = PReLU()(h_1)
        h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_1)
        h_1 = PReLU()(h_1)
        h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_1)+h
        h = PReLU()(h_1)
    h = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(h)
    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(16)(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model



def create_model_wo_ls_densenet(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h_2 =tf.keras.layers.concatenate([h, h_1], axis =-1)
        h_2 = PReLU()(h_2)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_3 = tf.keras.layers.concatenate([h, h_1, h_2], axis =-1)
        h_3 = PReLU()(h_3)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_3)
        h = PReLU()(h)
    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model

def create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = input_layer_1
    h_2 = input_layer_2

    l_2 = Conv2D(6, (3,3), activation=None, padding='same')
    h_2 = l_2(h_2)
        
    for i in range(len(filters)):
        if i == 0:
            h = l_2(h)
            h = PReLU()(h)
            h = BatchNormalization()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)

    h = Flatten()(h)
    h = PReLU()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(tf.cast(h,tf.float64))
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model


def create_model_wo_ls_peak_siamese(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    #h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)#input_layer_1+input_layer_2#tf.keras.layers.multiply([input_layer_1, input_layer_2])#
    h = BatchNormalization()(input_layer_1)
    h_2 = input_layer_2#BatchNormalization()(input_layer_2)
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    
    h = Dense(hidden_dim)(h)
    h = PReLU()(h)
    h_2 = Dense(hidden_dim, activation='sigmoid')(h_2)
    h = tf.keras.layers.concatenate([h, h_2])#h-h_2
        #h = h+h_2
    #h = h + tf.keras.layers.concatenate([h, h_2])
    h = PReLU()(h)
    h = Flatten()(h)

    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model



def create_model_prob_flipout_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)
    h = BatchNormalization()(h)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h_2 = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DFlipout(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h)
        h = PReLU()(h)

        h = tf.keras.layers.concatenate([h, h_2], axis =-1)

    h = Flatten()(h)
    h = tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model


def create_model_prob_rt_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)
    h = BatchNormalization()(h)
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h_2 = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same')(h)
        h = PReLU()(h)
        h = tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]),
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', strides=(2,2))(h)
        h = PReLU()(h)

        h = tf.keras.layers.concatenate([h, h_2], axis =-1)

    h = Flatten()(h)
    h = tfp.layers.DenseReparameterization(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model([input_layer_1, input_layer_2],h)

    return model

