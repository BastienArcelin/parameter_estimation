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
batch_size = 256

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

# With generator
#images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/blended_galaxies/random/'
images_dir = '/pbs/home/b/barcelin/sps_link/data/dc2_test/'#1_matching/'#deconv_conv_24.5/

list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training_24.5_v2/')) if x.startswith(os.path.join(images_dir,'training_24.5_v2/')+'img_noiseless_sample')]#mag_24.5
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation_24.5_v2/')) if x.startswith(os.path.join(images_dir,'validation_24.5_v2/')+'img_noiseless_sample')]#mag_24.5

if (sys.argv[1] == 'noiseless'):
    ################# With generators
    training_generator = generator.BatchGenerator_dc2_deconv_vae(bands,
                                        images_dir,
                                        list_of_samples, 
                                        total_sample_size=None,
                                        batch_size=batch_size, 
                                        trainval_or_test='training',
                                        do_norm=False,
                                        denorm = False,
                                        list_of_weights_e=None)

    validation_generator = generator.BatchGenerator_dc2_deconv_vae(bands,
                                        images_dir,
                                        list_of_samples_val, 
                                        total_sample_size=None,
                                        batch_size=batch_size, 
                                        trainval_or_test='validation',
                                        do_norm=False,
                                        denorm = False,
                                        list_of_weights_e=None)

    test_generator = generator.BatchGenerator_dc2_deconv_vae(bands, 
                                        images_dir,
                                        list_of_samples_val, 
                                        total_sample_size=None,
                                        batch_size=batch_size, 
                                        trainval_or_test='validation',
                                        do_norm=False,
                                        denorm = False,
                                        list_of_weights_e=None)

else:
    ################# With generators
    training_generator = generator.BatchGenerator_dc2_deconv_vae_noisy(bands,
                                        images_dir,
                                        list_of_samples, 
                                        total_sample_size=None,
                                        batch_size=batch_size, 
                                        trainval_or_test='training',
                                        do_norm=False,
                                        denorm = False,
                                        list_of_weights_e=None)

    validation_generator = generator.BatchGenerator_dc2_deconv_vae_noisy(bands,
                                        images_dir,
                                        list_of_samples_val, 
                                        total_sample_size=None,
                                        batch_size=batch_size, 
                                        trainval_or_test='validation',
                                        do_norm=False,
                                        denorm = False,
                                        list_of_weights_e=None)

    test_generator = generator.BatchGenerator_dc2_deconv_vae_noisy(bands, 
                                        images_dir,
                                        list_of_samples_val, 
                                        total_sample_size=None,
                                        batch_size=batch_size, 
                                        trainval_or_test='validation',
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
output_types = ((tf.float32,tf.float32), tf.float32)
output_shapes = ((tf.TensorShape([batch_size, 59, 59, nb_of_bands]),tf.TensorShape([batch_size, 59, 59, nb_of_bands])),
                tf.TensorShape([batch_size, 59, 59, nb_of_bands]))

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

#### Model definition
model_choice = str(sys.argv[5])#'full_prob_flipout'
# VAE with PSF input
if model_choice == 'with_psf':#create_model_wo_ls_resnet #create_model_wo_ls_peak_3 #create_model_3D
    net = model.create_model_wo_ls_peak_pooling_vae_3(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)  #create_model_wo_ls_peak_3
# VAE without PSF input
if model_choice == 'without_psf':
    net = model.create_model_wo_ls_peak_pooling_vae_4(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
net.summary()

if (sys.argv[1] == 'noisy'):
    # set the decoder as non trainable
    for i in range (len(net.layers[45:])):
        net.layers[45+i].trainable = False

#### Loss definition
model_loss = str(sys.argv[6])
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


net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
              loss=vae_loss,
              metrics = ['mse', 'acc',kl_metric],
              experimental_run_tf_function=False)


if (str(sys.argv[3]) == 'loading'):
    loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/'+str(sys.argv[4])+'/mse/'
    print(loading_path)
    latest = tf.train.latest_checkpoint(loading_path)
    net.load_weights(latest)


# Callbacks
saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/'+str(sys.argv[2])
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)#mse en TF2
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
#checkpointer_acc = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/acc/weights_noisy_v4.{epoch:02d}-{val_acc:.2f}.ckpt', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max', period=1)

callbacks = [checkpointer_mse, checkpointer_loss]#, checkpointer_acc]#, alpha_changer]#, alpha_changer]#, WandbCallback()]#, alpha_changer]


######## Train the network
## With dataset (faster than directly from generator)
hist = net.fit(training_generator, epochs=1000,#training_ds
                    steps_per_epoch=steps_per_epoch,
                    verbose=2,
                    shuffle=True,
                    callbacks = callbacks,
                    validation_data=validation_generator,#validation_ds
                    validation_steps=validation_steps)

saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/'+str(sys.argv[2])
net.save_weights(saving_path+'cp-{epoch:04d}.ckpt')

#### Plots
## REGENERER AVEC NOUVELLES IMAGES ET RENORMALISATION CORRECTE
loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/'+str(sys.argv[2])+'/mse/'#test_5
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)
test = test_generator.__getitem__(3)

training_data = test[0]#[0], test[0][1]]
training_labels = test[1]
#print(training_data.shape)
out = net([tf.cast(training_data[0], tf.float32), tf.cast(training_data[1], tf.float32)])# net(training_data) en TF2

#print('mean e2: '+str(np.mean(K.get_value(out.mean())[:,0]))+' mean e2: '+str(np.mean(K.get_value(out.mean())[:,1])))

fig,axes = plt.subplots(5,5, figsize =(20,20))
for i in range(5):
    f1 = axes[i,0].imshow(test[0][0][i,:,:,2], label='input gal')
    f2 = axes[i,1].imshow(test[0][1][i,:,:,2], label='input psf')
    f3 = axes[i,2].imshow(test[1][i,:,:,2], label='target')
    f4 = axes[i,3].imshow(K.get_value(out)[i,:,:,2], label='output')
    f5 = axes[i,4].imshow(test[1][i,:,:,2]-K.get_value(out)[i,:,:,2], label='residual')

    fig.colorbar(f1 ,ax=axes[i,0])
    fig.colorbar(f2 ,ax=axes[i,1])
    fig.colorbar(f3 ,ax=axes[i,2])
    fig.colorbar(f4 ,ax=axes[i,3])
    fig.colorbar(f5 ,ax=axes[i,4])

fig.savefig('full_prob/test_vae.png')


# fig = plt.figure()
# sns.distplot(K.get_value(out.mean())[:,1], bins = 20)# out.mean().numpy()
# sns.distplot(training_labels[:,1], bins = 20)
# fig.savefig('full_prob/test_distrib_e2.png')


# # fig = plt.figure()
# # sns.distplot(K.get_value(out.mean())[:,2], bins = 20)# out.mean().numpy()
# # sns.distplot(training_labels[:,2], bins = 20)
# # fig.savefig('full_prob/test_distrib_e3.png')

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# nb_of_points = 100
# axes[0].errorbar(training_labels[:nb_of_points,0], K.get_value(out.mean())[:nb_of_points,0], yerr = 2*K.get_value(out.stddev())[:nb_of_points,0],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
# x = np.linspace(-1,1)#(-1,1)#-0,5
# axes[0].plot(x, x)
# axes[0].legend()
# axes[0].set_ylim(-1,1)#(-1,1)#-1,1
# axes[0].set_title('$e1$')

# axes[1].errorbar(training_labels[:nb_of_points,1], K.get_value(out.mean())[:nb_of_points,1], yerr = 2*K.get_value(out.stddev())[:nb_of_points,1],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
# x = np.linspace(-1,1)#(-1,1)#-1,1
# axes[1].plot(x, x)
# axes[1].legend()
# axes[1].set_ylim(-1,1)#(-1,1)#-1,1
# axes[1].set_title('$e2$')

# axes[2].errorbar(training_labels[:nb_of_points,2], K.get_value(out.mean())[:nb_of_points,2], yerr = 2*K.get_value(out.stddev())[:nb_of_points,2],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
# x = np.linspace(-0.5,4)
# axes[2].plot(x, x)
# axes[2].legend()
# axes[2].set_ylim(-0.5,3.5)
# axes[2].set_title('$z$')

# fig.savefig('full_prob/test_train.png')

