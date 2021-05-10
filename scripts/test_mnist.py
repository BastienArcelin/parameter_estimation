import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn import preprocessing
from importlib import reload
import matplotlib as mpl
import scipy
import scipy.stats as stats

sys.path.insert(0,'tools_for_VAE/')
from tools_for_VAE import utils, vae_functions, generator, model, boxplot, plot

#%matplotlib inline
#%config InlineBackend.figure_format='retina'

import tensorflow.compat.v1 as tf1
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import util as tfp_layers_util
#from tensorboard.plugins.hparams import api as hp


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))




# Records the weights throughout the training process
weights_history = []
biases_history = []
other_history = []
other_2_history = []

weights_sec_history = []
biases_sec_history = []
other_sec_history = []
other_2_sec_history = []

weights_third_history = []
biases_third_history = []
other_third_history = []
other_2_third_history = []

weights_fourth_history = []
biases_fourth_history = []
other_fourth_history = []
other_2_fourth_history = []

# A custom callback
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
class MyCallback(keras.callbacks.Callback):
    def __init__(self):
        self.epoch =0
    def on_batch_end(self, batch, logs):
        # weights_std, weights_mean, _biases_std, _biases_mean, weights_std_2, weights_mean_2, _biases_std_2, _biases_mean_2, weights_std_3, weights_mean_3, _biases_std_3, _biases_mean_3, weights_std_4, weights_mean_4, _biases_std_4, _biases_mean_4, weights_std_5, weights_mean_5, _biases_std_5, _biases_mean_5, weights_std_6, weights_mean_6, _biases_std_6, _biases_mean_6= model.get_weights() 
        
        # weights_std = np.mean(K.get_value(weights_std))
        # weights_mean = np.mean(K.get_value(weights_mean))
        # _biases_std = np.mean(K.get_value(_biases_std))
        # _biases_mean = np.mean(K.get_value(_biases_mean))
        
        # weights_std_2 = np.mean(K.get_value(weights_std_2))
        # weights_mean_2 = np.mean(K.get_value(weights_mean_2))
        # _biases_std_2 = np.mean(K.get_value(_biases_std_2))
        # _biases_mean_2 = np.mean(K.get_value(_biases_mean_2))
        
        # weights_std_3 = np.mean(K.get_value(weights_std_3))
        # weights_mean_3 = np.mean(K.get_value(weights_mean_3))
        # _biases_std_3 = np.mean(K.get_value(_biases_std_3))
        # _biases_mean_3 = np.mean(K.get_value(_biases_mean_3))
        
        # weights_std_4 = np.mean(K.get_value(weights_std_4))
        # weights_mean_4 = np.mean(K.get_value(weights_mean_4))
        # _biases_std_4 = np.mean(K.get_value(_biases_std_4))
        # _biases_mean_4 = np.mean(K.get_value(_biases_mean_4))

        # weights_history.append(weights_std)
        # biases_history.append(weights_mean)
        # other_history.append(_biases_std)
        # other_2_history.append(_biases_mean)

        # weights_sec_history.append(weights_std_2)
        # biases_sec_history.append(weights_mean_2)
        # other_sec_history.append(_biases_std_2)
        # other_2_sec_history.append(_biases_mean_2)

        # weights_third_history.append(weights_std_3)
        # biases_third_history.append(weights_mean_3)
        # other_third_history.append(_biases_std_3)
        # other_2_third_history.append(_biases_mean_3)
        
        # weights_fourth_history.append(weights_std_4)
        # biases_fourth_history.append(weights_mean_4)
        # other_fourth_history.append(_biases_std_4)
        # other_2_fourth_history.append(_biases_mean_4)

        self.epoch +=1
        #print(self.epoch)
        if (self.epoch == 59) or (self.epoch ==59000):
            fig, axes = plt.subplots(1,2,figsize = (12,4))
            print('IN')
            density = []
            color = ['b', 'r', 'g', 'grey','black','y']
            for i in range (4):
                density = stats.gaussian_kde(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.mean()), (K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[1])))
                n, x, _ = axes[0].hist(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.mean()), (K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[1])), bins = 100, normed = True,label = 'weight '+str(i+1)+' mean', alpha = 0.3, color = color[i])
                axes[0].plot(x, density(x), color = color[i])
                axes[0].legend()

                density = stats.gaussian_kde(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.stddev()), (K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[1])))
                n, x, _ = axes[1].hist(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.stddev()), (K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[1])), bins = 100, normed = True ,label = 'weight '+str(i+1)+' stddev', alpha = 0.3, color = color[i])
                axes[1].plot(x, density(x), color = color[i])
                #axes[1].set_xscale('log')
                axes[1].legend()

            fig.savefig('mnist_test_hist_'+str(self.epoch)+'.png')
            #self.epoch = 0
        
callback = MyCallback()



# Weights initialization for posteriors
def get_posterior_fn():
    return tfp_layers_util.default_mean_field_normal_fn()
      #loc_initializer=tf1.initializers.he_normal(), 
      #untransformed_scale_initializer=tf1.initializers.random_normal(
      #    mean=-9, stddev=0.1)
      #)
# kernel divergence weight in loss
#HP_kl = hp.HParam('kl', hp.RealInterval(0., 1.))
def get_prior_fn():
    return tfp_layers_util.default_mean_field_normal_fn(
      #loc_initializer=tf1.initializers.he_normal(), 
      #untransformed_scale_initializer=tf1.initializers.random_normal(
      #    mean=-9, stddev=0.0001)
      )

tfd = tfp.distributions
kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p) / (60000))

model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # tfp.layers.DenseFlipout(1024, 
    #                          kernel_posterior_fn=get_posterior_fn(), 
    #                          kernel_divergence_fn = kernel_divergence_fn,
    #                          kernel_prior_fn = get_prior_fn(),
    #                          bias_posterior_fn = get_posterior_fn(),
    #                          bias_divergence_fn = kernel_divergence_fn,
    #                          activation='relu'),
    # tfp.layers.DenseFlipout(1024, 
    #                          kernel_posterior_fn=get_posterior_fn(), 
    #                          kernel_divergence_fn = kernel_divergence_fn,
    #                          kernel_prior_fn = get_prior_fn(),
    #                          bias_posterior_fn = get_posterior_fn(),
    #                          bias_divergence_fn = kernel_divergence_fn,
    #                          activation='relu'),
    tfp.layers.DenseFlipout(1024, 
                             kernel_posterior_fn=get_posterior_fn(), 
                             kernel_divergence_fn = kernel_divergence_fn, 
                             kernel_prior_fn = get_prior_fn(),
                             bias_posterior_fn = get_posterior_fn(),
                             bias_divergence_fn = kernel_divergence_fn,
                             bias_prior_fn = get_prior_fn(),
                             activation='relu'),
    tfp.layers.DenseFlipout(1024, 
                             kernel_posterior_fn=get_posterior_fn(), 
                             kernel_divergence_fn = kernel_divergence_fn, 
                             kernel_prior_fn = get_prior_fn(),
                             bias_posterior_fn = get_posterior_fn(),
                             bias_divergence_fn = kernel_divergence_fn,
                             bias_prior_fn = get_prior_fn(),
                             activation='relu'),
    tfp.layers.DenseFlipout(1024, 
                             kernel_posterior_fn=get_posterior_fn(), 
                             kernel_divergence_fn = kernel_divergence_fn, 
                             kernel_prior_fn = get_prior_fn(),
                             bias_posterior_fn = get_posterior_fn(),
                             bias_divergence_fn = kernel_divergence_fn,
                             bias_prior_fn = get_prior_fn(),
                             activation='relu'),
    
    tfp.layers.DenseFlipout(10, 
                             kernel_posterior_fn=get_posterior_fn(), 
                             kernel_divergence_fn = kernel_divergence_fn, 
                             kernel_prior_fn = get_prior_fn(),
                             bias_posterior_fn = get_posterior_fn(),
                             bias_divergence_fn = kernel_divergence_fn,
                             bias_prior_fn = get_prior_fn(),
                             activation=None),
    # tf.keras.layers.Dense(1024, activation = 'relu'),
    # tf.keras.layers.Dense(1024, activation = 'relu'),
    # tf.keras.layers.Dense(1024, activation = 'relu'),
    # tf.keras.layers.Dense(10, activation = None),
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Custom metrics
def kl_metric(y_true, y_pred):
    return K.sum(model.losses)

learning_rate = 1e-3
model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate) ,
              loss=loss_fn,
              metrics=['accuracy', kl_metric])

model.fit(x_train, y_train, epochs=5000, batch_size=1024,
          verbose=1, callbacks=[callback])


model.summary()


print('layer 1')
#print(K.get_value(model.layers[0].kernel_posterior.mean()))
#print(K.get_value(tf.nn.softplus(model.layers[0].kernel_posterior.stddev())))

print('layer 2')
print(np.mean(K.get_value(model.layers[1].kernel_posterior.mean())))
print(np.mean(K.get_value(tf.nn.softplus(model.layers[1].kernel_posterior.stddev()))))

print('layer 3')
print(np.mean(K.get_value(model.layers[2].kernel_posterior.mean())))
print(np.mean(K.get_value(tf.nn.softplus(model.layers[2].kernel_posterior.stddev()))))

print('layer 4')
print(np.mean(K.get_value(model.layers[3].kernel_posterior.mean())))
print(np.mean(K.get_value(tf.nn.softplus(model.layers[3].kernel_posterior.stddev()))))

print('layer 5')
print(np.mean(K.get_value(model.layers[4].kernel_posterior.mean())))
print(np.mean(K.get_value(tf.nn.softplus(model.layers[4].kernel_posterior.stddev()))))

fig, axes = plt.subplots(1,2,figsize = (12,4))

density = []
color = ['b', 'r', 'g', 'grey','black','y']
for i in range (4):
    density = stats.gaussian_kde(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.mean()), (K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[1])))
    n, x, _ = axes[0].hist(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.mean()), (K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.mean()).shape[1])), bins = 100, normed = True,label = 'weight '+str(i+1)+' mean', alpha = 0.3, color = color[i])
    axes[0].plot(x, density(x), color = color[i])
    axes[0].legend()

    density = stats.gaussian_kde(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.stddev()), (K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[1])))
    n, x, _ = axes[1].hist(np.reshape(K.get_value(model.layers[i+1].kernel_posterior.stddev()), (K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[0]*K.get_value(model.layers[i+1].kernel_posterior.stddev()).shape[1])), bins = 100, normed = True ,label = 'weight '+str(i+1)+' stddev', alpha = 0.3, color = color[i])
    axes[1].plot(x, density(x), color = color[i])
    #axes[1].set_xscale('log')
    axes[1].legend()

fig.savefig('mnist_test_hist.png')

# fig, axes = plt.subplots(4,4,figsize = (20,20))

# axes[0,0].plot(K.get_value(tf.nn.softplus(weights_history)), label = 'weight 1 std')
# axes[0,1].plot(biases_history, label = 'weight 1 mean')
# axes[0,2].plot(other_history, label = 'bias 1 std')
# axes[0,3].plot(other_2_history, label = 'bias 1 mean')

# axes[1,0].plot(weights_sec_history, label = 'weight 2 std')
# axes[1,1].plot(biases_sec_history, label = 'weight 2 mean')
# axes[1,2].plot(other_sec_history, label = 'bias 2 std')
# axes[1,3].plot(other_2_sec_history, label = 'bias 2 mean')

# axes[2,0].plot(weights_third_history, label = 'weight 3 std')
# axes[2,1].plot(biases_third_history, label = 'weight 3 mean')
# axes[2,2].plot(other_third_history, label = 'bias 3 std')
# axes[2,3].plot(other_2_third_history, label = 'bias 3 mean')

# axes[3,0].plot(weights_fourth_history, label = 'weight 4 std')
# axes[3,1].plot(biases_fourth_history, label = 'weight 4 mean')
# axes[3,2].plot(other_fourth_history, label = 'bias 4 std')
# axes[3,3].plot(other_2_fourth_history, label = 'bias 4 mean')

# plt.legend()
# fig.savefig('mnist_test.png')