import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# TFP imports
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.layers import util as tfp_layers_util
import tensorflow.compat.v1 as tf1

def ktied_loc_scale_fn(
    is_singular=False,
    loc_initializer=tf1.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf1.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None,
    tying_rank=2):
  
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    """Creates `loc`, `scale` parameters."""
    print(shape)
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=loc_initializer,
        regularizer=loc_regularizer,
        constraint=loc_constraint,
        dtype=dtype,
        trainable=trainable)
    if is_singular:
      return loc, None

    stddev_mean = untransformed_scale_initializer.mean
    stddev_stddev = untransformed_scale_initializer.stddev
    
    tied_stddev_mean = 0.5 * np.log(np.exp(stddev_mean) / tying_rank)
    tied_stddev_init = tf.keras.initializers.TruncatedNormal(mean=tied_stddev_mean, 
                                                      stddev=stddev_stddev)

    untransformed_scal_u = add_variable_fn(
        name=name + '_untransformed_scale_u',
        shape=(shape[0], tying_rank),
        initializer=tied_stddev_init,
        regularizer=untransformed_scale_regularizer,
        constraint=untransformed_scale_constraint,
        dtype=dtype,
        trainable=trainable)
    
    untransformed_scal_v = add_variable_fn(
        name=name + '_untransformed_scale_v',
        shape=(shape[1], tying_rank),
        initializer=tied_stddev_init,
        regularizer=untransformed_scale_regularizer,
        constraint=untransformed_scale_constraint,
        dtype=dtype,
        trainable=trainable)
    scale = tf.matmul(tf.exp(untransformed_scal_u), tf.transpose(tf.exp(untransformed_scal_v)))
    return loc, scale
  return _fn


def ktied_mean_field_normal_fn(
    is_singular=False,
    loc_initializer=tf1.initializers.random_normal(stddev=0.1),
    untransformed_scale_initializer=tf1.initializers.random_normal(
        mean=-3., stddev=0.1),
    loc_regularizer=None,
    untransformed_scale_regularizer=None,
    loc_constraint=None,
    untransformed_scale_constraint=None):
  
  loc_scale_fn = ktied_loc_scale_fn(
      is_singular=is_singular,
      loc_initializer=loc_initializer,
      untransformed_scale_initializer=untransformed_scale_initializer,
      loc_regularizer=loc_regularizer,
      untransformed_scale_regularizer=untransformed_scale_regularizer,
      loc_constraint=loc_constraint,
      untransformed_scale_constraint=untransformed_scale_constraint)
  def _fn(dtype, shape, name, trainable, add_variable_fn):
    loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
    if scale is None:
      dist = tfd.Deterministic(loc=loc)
    else:
      dist = tfd.Normal(loc=loc, scale=scale)
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    print(batch_ndims)
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
  return _fn


def get_ktied_posterior_fn():
  return ktied_mean_field_normal_fn(
      loc_initializer=tf1.initializers.he_normal(), 
      untransformed_scale_initializer=tf1.initializers.random_normal(
          mean=-9.0, stddev=0.1)
      )