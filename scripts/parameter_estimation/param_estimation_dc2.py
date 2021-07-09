#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
import galsim

import tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import backend as K
tfd = tfp.distributions

sys.path.insert(0,'../../scripts/tools_for_VAE/')
import tools_for_VAE.layers as layers
from tools_for_VAE import utils, vae_functions, generator, model
from tools_for_VAE.callbacks import changeAlpha


import wandb
from wandb.keras import WandbCallback
#wandb.init()
######## Parameters
nb_of_bands = 6#1
batch_size = 512

input_shape = (59, 59, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 2
filters = [32,64,128,256]#,128,256]#[32,64,128,256]#[8,16,32,64]#[32,64,128,256]#,128]#[32,64,128,256,512]
kernels = [3,3,3,3]#[3,3,3,3]##,3]

conv_activation = None
dense_activation = None

steps_per_epoch = int(100000/batch_size)
validation_steps = int(20000/batch_size)

bands = [0,1,2,3,4,5]

saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/'+str(sys.argv[2])


#### Model definition

model_choice = str(sys.argv[1])
# Without latent space
if model_choice == 'wo_ls_fft_moments':
    net = model.create_model_wo_ls_peak_pooling_fft_moments(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None) 
if model_choice == 'wo_ls_fft':
    net = model.create_model_wo_ls_peak_pooling_fft_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
if model_choice == 'wo_ls_dense':
    net = model.create_model_wo_ls_peak_pooling_dense(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
if model_choice == 'wo_ls_no_psf':
    net = model.create_model_wo_ls_peak_pooling_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
if model_choice == 'wo_ls':
    net = model.create_model_wo_ls_peak_pooling(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
if model_choice == 'wo_ls_concat_one':
    net = model.create_model_wo_ls_peak_pooling_concat_one(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    #net = model.create_model_shear(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
if model_choice == 'full_prob_flipout':
    net = model.create_model_prob_flipout_peak_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
if model_choice == 'cyrille':
    net = model.create_model_cyrille(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
if model_choice == 'no_psf_dense':
    net = model.create_model_no_psf_dense(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)

if model_choice == 'redshift':
    net = model.create_model_wo_ls_peak_pooling_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, 1, conv_activation=None, dense_activation=None)
if model_choice == 'redshift_bnn':
    net = model.create_model_prob_flipout_peak_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, 1, conv_activation=None, dense_activation=None)

if model_choice == 'redshift_ellipticity':
    net = model.create_model_wo_ls_peak_pooling_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, 3, conv_activation=None, dense_activation=None)
if model_choice == 'redshift_ellipticity_bnn':
    net = model.create_model_prob_flipout_peak_no_psf(input_shape, latent_dim, hidden_dim, filters, kernels, 3, conv_activation=None, dense_activation=None)


net.summary()
#net.layers[-1].trainable = False

#### Loss definition
alpha = K.variable(1.)

if model_choice == 'full_prob_rt' or model_choice == 'full_prob_flipout':
    kl = sum(net.losses)
    def loss(x, dists):
        nll = -dists.log_prob(x)
        print(nll)
        kl = sum(net.losses)
        print(kl)
        return nll + kl, collections.namedtuple('loss','nll,kl')(nll, kl)

    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)+ kl *(K.get_value(alpha)-1)

else:
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)


# Custom metrics
def kl_metric(y_true, y_pred):
    return K.sum(net.losses)
mse = tf.keras.losses.MeanSquaredError()
hinge = tf.keras.losses.Hinge()
maepercent = tf.keras.losses.MeanAbsolutePercentageError()
mae = tf.keras.losses.MeanAbsoluteError()
mselog = tf.keras.losses.MeanSquaredLogarithmicError()

net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4),
              loss=negative_log_likelihood,
              metrics = ['mse', 'acc',kl_metric],
              experimental_run_tf_function=False)



# Data generator
#images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/blended_galaxies/random/'
images_dir = '/pbs/home/b/barcelin/sps_link/data/dc2_test/24.5/'#1_matching/'#deconv_conv_24.5/

## The difference between noiseless and noisy case depends on the size of the step for the increment of noisy data.
if (int(sys.argv[5]) == None):
    print('in noiseless')
    step_size = 10000

else:
    print('Noisy')
    step_size = int(sys.argv[5])
    print(step_size)


list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training/')) if x.startswith(os.path.join(images_dir,'training/')+'img_cropped_sample_')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation/')) if x.startswith(os.path.join(images_dir,'validation/')+'img_cropped_sample_')]
#list_of_samples_test = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('img_sample.npy')]#mag_24.5
print(list_of_samples)
training_generator = generator.BatchGenerator_redshift_ellipticity(bands,#BatchGenerator_dc2_deconv_noisy_2
                                    images_dir,
                                    list_of_samples, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None,
                                    net = net,
                                    saving_path = saving_path,
                                    prop = 10,
                                    step_size = step_size)#BatchGenerator_redshift

validation_generator = generator.BatchGenerator_redshift_ellipticity(bands,
                                    images_dir,
                                    list_of_samples_val, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None,
                                    net = net,
                                    saving_path = saving_path,
                                    prop = 10,
                                    step_size = step_size)#BatchGenerator_dc2_reconvolution

test_generator = generator.BatchGenerator_redshift_ellipticity(bands, 
                                    images_dir,
                                    list_of_samples_val, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None,
                                    net = net,
                                    saving_path = saving_path,
                                    prop = 0,
                                    step_size = step_size)#BatchGenerator_dc2_reconvolution


print('construction OK')



if (str(sys.argv[3]) == 'loading'):
    print('in loading')
    loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/'+str(sys.argv[4])+'/loss/'
    print(loading_path)
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)


# Callbacks
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)#mse en TF2
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_acc = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/acc/weights_noisy_v4.{epoch:02d}-{val_acc:.2f}.ckpt', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max', period=1)

alpha_changer = changeAlpha(alpha, net,negative_log_likelihood, kl_metric)

from tensorflow.keras.callbacks import Callback
#print(saving_path)
class save_model_step(Callback):
    def __init__(self, network, save_path, step_siz):
        self.epoch = 0
        self.step_siz = step_siz
        self.save_path = save_path
        self.save_path = str(self.save_path)+'/end_step/'
        self.network = network
    
    def on_epoch_end(self, network, save_path):
        if (self.epoch == self.step_siz):
            self.epoch =0
            print(self.save_path)
            self.network.save_weights(self.save_path+'cp-'+str(self.epoch)+'.ckpt')
        self.epoch +=1
        #print(self.epoch)

cb = save_model_step(net, saving_path, step_size)

callbacks = [checkpointer_mse, checkpointer_loss,checkpointer_acc, cb]#, checkpointer_acc]#, alpha_changer]#, alpha_changer]#, WandbCallback()]#, alpha_changer]


#print(saving_path)
#print('debut entrainement')
######## Train the network
## With dataset (faster than directly from generator)
hist = net.fit(training_generator, epochs=2000,#training_ds
                    steps_per_epoch=steps_per_epoch,
                    verbose=2,
                    shuffle=True,
                    callbacks = callbacks,
                    validation_data=validation_generator,#validation_ds
                    validation_steps=validation_steps)

net.save_weights(saving_path+'cp-{epoch:04d}.ckpt')

#### Plots
## REGENERER AVEC NOUVELLES IMAGES ET RENORMALISATION CORRECTE
loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/'+str(sys.argv[2])+'/loss/'#test_5
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)
test = test_generator.__getitem__(3)

training_data = test[0]#[0], test[0][1]]
training_labels = test[1]
#print(training_data.shape)
out = net([tf.cast(training_data[0], tf.float32), tf.cast(training_data[1], tf.float32)])# net(training_data) en TF2

print('mean e2: '+str(np.mean(K.get_value(out.mean())[:,0]))+' mean e2: '+str(np.mean(K.get_value(out.mean())[:,1])))

fig = plt.figure()
sns.distplot(K.get_value(out.mean())[:,0], bins = 20)# out.mean().numpy()
sns.distplot(training_labels[:,0], bins = 20)
fig.savefig('full_prob/test_distrib_e1.png')


fig = plt.figure()
sns.distplot(K.get_value(out.mean())[:,1], bins = 20)# out.mean().numpy()
sns.distplot(training_labels[:,1], bins = 20)
fig.savefig('full_prob/test_distrib_e2.png')



# fig = plt.figure()
# sns.distplot(K.get_value(out.mean())[:,2], bins = 20)# out.mean().numpy()
# sns.distplot(training_labels[:,2], bins = 20)
# fig.savefig('full_prob/test_distrib_e3.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

nb_of_points = 100
axes[0].errorbar(training_labels[:nb_of_points,0], K.get_value(out.mean())[:nb_of_points,0], yerr = 2*K.get_value(out.stddev())[:nb_of_points,0],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
x = np.linspace(-1,1)#(-1,1)#-0,5
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)#(-1,1)#-1,1
axes[0].set_title('$e1$')

axes[1].errorbar(training_labels[:nb_of_points,1], K.get_value(out.mean())[:nb_of_points,1], yerr = 2*K.get_value(out.stddev())[:nb_of_points,1],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
x = np.linspace(-1,1)#(-1,1)#-1,1
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)#(-1,1)#-1,1
axes[1].set_title('$e2$')

# axes[2].errorbar(training_labels[:nb_of_points,2], K.get_value(out.mean())[:nb_of_points,2], yerr = 2*K.get_value(out.stddev())[:nb_of_points,2],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
# x = np.linspace(-0.5,4)
# axes[2].plot(x, x)
# axes[2].legend()
# axes[2].set_ylim(-0.5,3.5)
# axes[2].set_title('$z$')

fig.savefig('full_prob/test_train.png')


##################################
# out = net([tf.cast(training_data[0], tf.float32), tf.cast(training_data[1], tf.float32)])
# fig = plt.figure()
# out = K.get_value(out)
# print(out.shape)
# sns.distplot(out[:,0], bins = 20, label = 'output')
# sns.distplot(training_labels[:,0], bins = 20, label = 'input')
# fig.savefig('full_prob/test_distrib_e1.png')


# fig = plt.figure()
# sns.distplot(out[:,1], bins = 20)
# sns.distplot(training_labels[:,1], bins = 20)
# fig.savefig('full_prob/test_distrib_e2.png')


# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].plot(training_labels[:,0], out[:,0], '.', label = 'mean')
# x = np.linspace(-1,1)
# axes[0].plot(x, x)
# axes[0].legend()
# axes[0].set_ylim(-1,1)
# axes[0].set_title('$e1$')

# axes[1].plot(training_labels[:,1], out[:,1], '.', label = 'mean')
# x = np.linspace(-1,1)
# axes[1].plot(x, x)
# axes[1].legend()
# axes[1].set_ylim(-1,1)
# axes[1].set_title('$e2$')

# # axes[2].plot(y[:,2], out[:,2], '.', label = 'mean')
# # x = np.linspace(0,4)
# # axes[2].plot(x, x)
# # axes[2].legend()
# # axes[2].set_ylim(-1,5.5)
# # axes[2].set_title('$z$')

# fig.savefig('full_prob/test_train.png')