#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from importlib import reload

import tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

tfd = tfp.distributions

sys.path.insert(0,'../../scripts/tools_for_VAE/')
import tools_for_VAE.layers as layers
from tools_for_VAE import utils, vae_functions, generator, model
from tools_for_VAE.callbacks import changeAlpha
import tensorflow.keras as keras
from tensorflow.keras import backend as K


######## Parameters
nb_of_bands = 6
batch_size = 256

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 3
filters = [32, 64, 128, 256]#, 512]
kernels = [3,3,3,3]#, 3]

conv_activation = None
dense_activation = None

steps_per_epoch = 32
validation_steps = 8

bands = [4,5,6,7,8,9]

# With generator
images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/blended_galaxies/random/'

list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]
list_of_samples_test = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
print(list_of_samples_test)

training_generator = generator.BatchGenerator_random_coord(bands,
                                    images_dir,
                                    list_of_samples, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

validation_generator = generator.BatchGenerator_random_coord(bands,
                                    images_dir,
                                    list_of_samples_val, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

test_generator = generator.BatchGenerator_random_coord(bands, 
                                    images_dir,
                                    list_of_samples_test, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='test',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)


# NEW: Wrap the generator.BatchGenerator objects in a generator-style function
# which we can then pass to tf.data.Dataset.from_generator()
# (One per train/val/test dataset at the moment, but could be refactored for neatness!)
def training_batch_generator():
    multi_enqueuer = keras.utils.OrderedEnqueuer(training_generator,
                                                use_multiprocessing=False)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_x, batch_y = next(multi_enqueuer.get())
        yield batch_x, batch_y

def validation_batch_generator():
    multi_enqueuer = keras.utils.OrderedEnqueuer(validation_generator,
                                                    use_multiprocessing=False)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_x, batch_y = next(multi_enqueuer.get())
        yield batch_x, batch_y

def test_batch_generator():
    multi_enqueuer = keras.utils.OrderedEnqueuer(test_generator,
                                                    use_multiprocessing=False)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_x, batch_y = next(multi_enqueuer.get())
        yield batch_x, batch_y

# Recommended to specify the expected output shapes and types here
# output_types = ((tf.float32, tf.float32), tf.float32)
# output_shapes = ((tf.TensorShape([batch_size, 64, 64, nb_of_bands]),tf.TensorShape([batch_size, 64, 64, nb_of_bands])),
#                     tf.TensorShape([batch_size, 3]))
output_types = ((tf.float32,tf.float32), tf.float32)
output_shapes = ((tf.TensorShape([batch_size, 64, 64, nb_of_bands]),tf.TensorShape([batch_size, 64, 64, nb_of_bands])),
                tf.TensorShape([batch_size, 3]))

training_ds = tf.data.Dataset.from_generator(training_batch_generator,
                                                output_types=output_types,
                                                output_shapes=output_shapes).repeat()

validation_ds = tf.data.Dataset.from_generator(validation_batch_generator,
                                                output_types=output_types,
                                                output_shapes=output_shapes).repeat()

test_ds = tf.data.Dataset.from_generator(test_batch_generator,
                                            output_types=output_types,
                                            output_shapes=output_shapes).repeat()

print('construction OK')
#print('construction OK: '+str((iter(test_ds))))
#### Model definition
model_choice = 'wo_ls'
# With latent space
if model_choice == 'ls':
    net = model.create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Without latent space
if model_choice == 'wo_ls':
    net = model.create_model_wo_ls_with_coord_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Full probabilistic model with reparametrization trick
if model_choice == 'full_prob_rt':
    net = model.create_model_full_prob_rt(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Full probabilistic model with flipout
if model_choice == 'full_prob_flipout':
    net = model.create_model_full_prob_flipout(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
net.summary()

#### Loss definition
alpha = K.variable(0.)

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

net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
              loss=negative_log_likelihood , metrics = ['mse', 'acc', kl_metric], experimental_run_tf_function=False)


# if str(sys.argv[3]).lower() == 'true':
loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_coord_4/loss/'#test_coord_2/loss/  # test_coord_1/loss/
print(loading_path)
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)


# Callbacks
saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_coord_4'
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)#mse en TF2
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_acc = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/acc/weights_noisy_v4.{epoch:02d}-{val_acc:.2f}.ckpt', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max', period=1)

alpha_changer = changeAlpha(alpha, net,negative_log_likelihood, kl_metric)

callbacks = [checkpointer_mse, checkpointer_loss, checkpointer_acc]#, alpha_changer]


######## Train the network
## With dataset (faster than directly from generator)
hist = net.fit(training_ds, epochs=100,
                    steps_per_epoch=steps_per_epoch,
                    verbose=1,
                    shuffle=True,
                    callbacks = callbacks,
                    validation_data=validation_ds,
                    validation_steps=validation_steps)

## Directly from generator
# hist = net.fit_generator(training_generator, #training_generator
#           epochs=10, 
#           steps_per_epoch=steps_per_epoch,
#           verbose=1,
#           shuffle=True,
#           validation_data=validation_generator, #validation_generator
#           validation_steps=validation_steps,
#           callbacks= callbacks,
#           workers=0,
#           use_multiprocessing = True)

saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_coord_4/'
net.save_weights(saving_path+'cp-{epoch:04d}.ckpt')


#### Plots
# training

## REGENERER AVEC NOUVELLES IMAGES ET RENORMALISATION CORRECTE
loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_coord_4/mse/'#test_5
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)
test = test_generator.__getitem__(2)

training_data = test[0]#[0], test[0][1]]
training_labels = test[1]
#print(training_data.shape)
out = net([tf.cast(training_data[0], tf.float32), tf.cast(training_data[1], tf.float32)])# net(training_data) en TF2

fig = plt.figure()
sns.distplot(K.get_value(out.mean())[:,0], bins = 20)# out.mean().numpy()
sns.distplot(training_labels[:,0], bins = 20)
fig.savefig('full_prob/test_distrib_e1.png')


fig = plt.figure()
sns.distplot(K.get_value(out.mean())[:,1], bins = 20)# out.mean().numpy()
sns.distplot(training_labels[:,1], bins = 20)
fig.savefig('full_prob/test_distrib_e2.png')


fig = plt.figure()
sns.distplot(K.get_value(out.mean())[:,2], bins = 20)# out.mean().numpy()
sns.distplot(training_labels[:,2], bins = 20)
fig.savefig('full_prob/test_distrib_e3.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

nb_of_points = 100
axes[0].errorbar(training_labels[:nb_of_points,0], K.get_value(out.mean())[:nb_of_points,0], yerr = 2*K.get_value(out.stddev())[:nb_of_points,0],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
#axes[0].plot(training_labels[:,0], K.get_value(out.mean())[:,0], '.', label = 'mean')# out.mean().numpy()
#axes[0].plot(training_labels[:,0], K.get_value(out.mean())[:,0]+ 2*K.get_value(out.stddev())[:,0], '+', label = 'mean + 2stddev')# out.mean().numpy()
#axes[0].plot(training_labels[:,0], K.get_value(out.mean())[:,0]- 2*K.get_value(out.stddev())[:,0], '+', label = 'mean - 2stddev')# out.mean().numpy()
x = np.linspace(-1,1)#-0,5
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)#-1,1
axes[0].set_title('$e1$')

axes[1].errorbar(training_labels[:nb_of_points,1], K.get_value(out.mean())[:nb_of_points,1], yerr = 2*K.get_value(out.stddev())[:nb_of_points,1],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
# axes[1].plot(training_labels[:,1], K.get_value(out.mean())[:,1], '.', label = 'mean')# out.mean().numpy()
# axes[1].plot(training_labels[:,1], K.get_value(out.mean())[:,1]+ 2*K.get_value(out.stddev())[:,1], '+', label = 'mean + 2stddev')# out.mean().numpy()
# axes[1].plot(training_labels[:,1], K.get_value(out.mean())[:,1]- 2*K.get_value(out.stddev())[:,1], '+', label = 'mean - 2stddev')# out.mean().numpy()
x = np.linspace(-1,1)#-1,1
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)#-1,1
axes[1].set_title('$e2$')

axes[2].errorbar(training_labels[:nb_of_points,2], K.get_value(out.mean())[:nb_of_points,2], yerr = 2*K.get_value(out.stddev())[:nb_of_points,2],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
# axes[2].plot(training_labels[:,2], K.get_value(out.mean())[:,2], '.', label = 'mean')# out.mean().numpy()
# axes[2].plot(training_labels[:,2], K.get_value(out.mean())[:,2]+ 2*K.get_value(out.stddev())[:,2], '+', label = 'mean + 2stddev')# out.mean().numpy()
# axes[2].plot(training_labels[:,2], K.get_value(out.mean())[:,2]- 2*K.get_value(out.stddev())[:,2], '+', label = 'mean - 2stddev')# out.mean().numpy()
x = np.linspace(-3,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-3,3)
axes[2].set_title('$z$')

fig.savefig('full_prob/test_train.png')

