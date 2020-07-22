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
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

import tensorflow as tf
tfd = tfp.distributions

sys.path.insert(0,'../../scripts/tools_for_VAE/')
from tools_for_VAE import ktied_distribution

def create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
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
          mean=-9.0, stddev=0.1)
      )
# kernel divergence weight in loss
kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p) / (512*32))

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
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    # Encoding part
    h = tf.keras.layers.concatenate([input_layer_1, tf.keras.layers.multiply([input_layer_1, input_layer_2])], axis=-1)#input_layer_1+input_layer_2#tf.keras.layers.multiply([input_layer_1, input_layer_2])#
    h = BatchNormalization()(h)
    h_2 = input_layer_2#BatchNormalization()(input_layer_2)
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)#'sigmoid'
        h_2 = PReLU()(h_2)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)#'sigmoid'
        h_2 = PReLU()(h_2)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)

        h = tf.keras.layers.concatenate([h, h_2], axis =-1)
        #h = tf.keras.layers.concatenate([h, tf.keras.layers.multiply([h, h_2])], axis=-1)
    #h = h + tf.keras.layers.multiply([h, h_2])
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


# For multiple outputs

def create_model_wo_ls_multi(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    tfd = tfp.distributions
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
                            reinterpreted_batch_ndims=1)

    input_layer = Input(shape=(input_shape)) 

    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        #h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        #h = PReLU()(h_1)
        #h = h_1 + Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        #h = PReLU()(h_2)
        #h = h_2 + Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    activation=None)(h)

    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    # h_1 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
    #                                 activation=None)(h)
    # h_1 = tfp.layers.MultivariateNormalTriL(final_dim)(h_1)
    # h_2 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
    #                                 activation=None)(h)
    # h_2 = tfp.layers.MultivariateNormalTriL(final_dim)(h_2)
    # h_3 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
    #                                 activation=None)(h)
    # h_3 = tfp.layers.MultivariateNormalTriL(final_dim)(h_3)

    #model = Model(input_layer,[h_1,h_2,h_3])
    model = Model(input_layer, h)
    return model

def create_model_wo_ls_multi_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    tfd = tfp.distributions
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
                            reinterpreted_batch_ndims=1)

    input_layer = Input(shape=(input_shape)) 

    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        #h_1 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        #h = PReLU()(h_1)
        #h = h_1 + Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        #h = PReLU()(h_2)
        #h = h_2 + Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    #h = PReLU()(h)
    #h = Dense(256)(h)
    #h = PReLU()(h)
    #h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
    #                               activation=None)(h)
    #h = PReLU()(h)
    #h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    h_1 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    activation=None)(h)
    h_1 = tfp.layers.MultivariateNormalTriL(final_dim)(h_1)
    h_2 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    activation=None)(h)
    h_2 = tfp.layers.MultivariateNormalTriL(final_dim)(h_2)
    h_3 = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), 
                                    activation=None)(h)
    h_3 = tfp.layers.MultivariateNormalTriL(final_dim)(h_3)

    model = Model(input_layer,[h_1,h_2,h_3])
    #model = Model(input_layer, h)
    return model



