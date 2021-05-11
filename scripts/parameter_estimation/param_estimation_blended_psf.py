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
batch_size = 128

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 2
filters = [32,64,128,256]#,512]
kernels = [5,4,3,3]#,3]

conv_activation = None
dense_activation = None

steps_per_epoch = int(10000/batch_size)
validation_steps = int(2000/batch_size)

bands = [4,5,6,7,8,9]#


# With generator
<<<<<<< HEAD
images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/blended_galaxies/random/'
#images_dir = '/pbs/home/b/barcelin/sps_link/data/psf_change/1_6_test/'

list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')][:4]#training
list_of_samples_val = [[x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')][0]]#validation
list_of_samples_test = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]

########### Directly loading data
# list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
# list_of_samples_labels = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.csv')]

# data = np.load(list_of_samples[0], mmap_mode = 'c')
# data_label = pd.read_csv(list_of_samples_labels[0])
# shifts = np.load(images_dir+'training/shifts/'+list_of_samples[0][-38:].replace('images.npy','shifts.npy'))

# # data_label.info()
# # ### Create samples
# nb_train = 2000
# x_train_1 = tf.transpose(data[:nb_train,1], perm= [0,2,3,1])[:,:,:,4:]
# x_train_2 = np.zeros((nb_train,64,64,6))
# print('Start generating PSF images')
# for i in range (nb_train):
#     z = np.random.random_integers(data_label['nb_blended_gal'][i])
#     psf = PSF_lsst.shift((shifts[i,z-1][0],shifts[i,z-1][1]))
#     temp_img = galsim.ImageF(img_size, img_size, scale=pixel_scale_lsst)
#     psf.drawImage(image=temp_img)
#     for m in range(6):
#         x_train_2[i,:,:,m]=temp_img.array.data
# print('Generation of PSF images ended')
# x_train = [x_train_1, x_train_2]
# y_train = np.zeros((nb_train,3))
# for i in range (nb_train):
#     if data_label['nb_blended_gal'][i]==1:
#         y_train[i,0] = data_label['e1_0'][i]
#         y_train[i,1] = data_label['e2_0'][i]
#         y_train[i,2] = data_label['redshift_0'][i]
#     else:
#         y_train[i,0] = data_label['e1_1'][i]
#         y_train[i,1] = data_label['e2_1'][i]
#         y_train[i,2] = data_label['redshift_1'][i]
# y_train = tf.convert_to_tensor(y_train)

# print(x_train[0].shape, x_train[1].shape, y_train.shape)

# nb_val = 300
# x_val_1 = tf.transpose(data[nb_train:nb_val+nb_train,1], perm= [0,2,3,1])[:,:,:,4:]
# x_val_2 = np.zeros((nb_val,64,64,6))
# print('Start generating PSF images validation')
# for i in range (nb_train, nb_val):
#     z = np.random.random_integers(data_label['nb_blended_gal'][i])
#     psf = PSF_lsst.shift((shifts[i,z-1][0],shifts[i,z-1][1]))
#     temp_img = galsim.ImageF(img_size, img_size, scale=pixel_scale_lsst)
#     psf.drawImage(image=temp_img)
#     for m in range(6):
#         x_val_2[i,:,:,m]=temp_img.array.data
# print('Generation of PSF images for validation ended')
# x_val = [x_val_1, x_val_2]
# y_val = np.zeros((nb_val,3))
# for i in range (nb_train, nb_val):
#     if data_label['nb_blended_gal'][i]==1:
#         y_val[i,0] = data_label['e1_0'][i]
#         y_val[i,1] = data_label['e2_0'][i]
#         y_val[i,2] = data_label['redshift_0'][i]
#     else:
#         y_val[i,0] = data_label['e1_1'][i]
#         y_val[i,1] = data_label['e2_1'][i]
#         y_val[i,2] = data_label['redshift_1'][i]
# y_val = tf.convert_to_tensor(y_val)





################# With generators
training_generator = generator.BatchGenerator_random_coord_psf(bands,
                                    images_dir,
                                    list_of_samples, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

validation_generator = generator.BatchGenerator_random_coord_psf(bands,
                                    images_dir,
                                    list_of_samples_val, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

test_generator = generator.BatchGenerator_random_coord_psf(bands, 
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
                tf.TensorShape([batch_size, final_dim]))

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

model_choice = 'wo_ls'
# Without latent space
if model_choice == 'wo_ls':
    net = model.create_model_wo_ls_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)  #create_model_wo_ls_peak_3
# Full probabilistic model with reparametrization trick
if model_choice == 'full_prob_rt':
    net = model.create_model_prob_rt_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Full probabilistic model with flipout
if model_choice == 'full_prob_flipout':
    net = model.create_model_prob_flipout_peak(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
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
              loss=negative_log_likelihood , 
              metrics = ['mse', 'acc', kl_metric],
              experimental_run_tf_function=False)



loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_recheck_OK/loss/'#test_coord_2/loss/  # test_coord_1/loss/
print(loading_path)
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)


# Callbacks
saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_recheck_OK/'#test_coord_psf_4/'
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)#mse en TF2
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_acc = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/acc/weights_noisy_v4.{epoch:02d}-{val_acc:.2f}.ckpt', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max', period=1)

alpha_changer = changeAlpha(alpha, net,negative_log_likelihood, kl_metric)

callbacks = [checkpointer_mse, checkpointer_loss, checkpointer_acc]#, alpha_changer]#, WandbCallback()]#, alpha_changer]


######## Train the network
## With dataset (faster than directly from generator)
hist = net.fit(training_ds, epochs=100,
                    steps_per_epoch=steps_per_epoch,
                    verbose=1,
                    shuffle=True,
                    callbacks = callbacks,
                    validation_data=validation_ds,
                    validation_steps=validation_steps)

## Directly loading data
print(validation_steps)
# hist = net.fit(x_train, y_train,
#           batch_size = batch_size, 
#           epochs=100, 
#           steps_per_epoch=steps_per_epoch,
#           verbose=1,
#           shuffle=True,
#           validation_data=(x_val,y_val),
#           validation_steps=validation_steps,
#           callbacks= callbacks)

saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_recheck_OK/'
net.save_weights(saving_path+'cp-{epoch:04d}.ckpt')


#### Plots
# training

## REGENERER AVEC NOUVELLES IMAGES ET RENORMALISATION CORRECTE
loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_recheck_OK/loss/'#test_5
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


# fig = plt.figure()
# sns.distplot(K.get_value(out.mean())[:,2], bins = 20)# out.mean().numpy()
# sns.distplot(training_labels[:,2], bins = 20)
# fig.savefig('full_prob/test_distrib_e3.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

nb_of_points = 100
axes[0].errorbar(training_labels[:nb_of_points,0], K.get_value(out.mean())[:nb_of_points,0], yerr = 2*K.get_value(out.stddev())[:nb_of_points,0],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
x = np.linspace(-1,1)#-0,5
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)#-1,1
axes[0].set_title('$e1$')

axes[1].errorbar(training_labels[:nb_of_points,1], K.get_value(out.mean())[:nb_of_points,1], yerr = 2*K.get_value(out.stddev())[:nb_of_points,1],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
x = np.linspace(-1,1)#-1,1
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)#-1,1
axes[1].set_title('$e2$')

# axes[2].errorbar(training_labels[:nb_of_points,2], K.get_value(out.mean())[:nb_of_points,2], yerr = 2*K.get_value(out.stddev())[:nb_of_points,2],  fmt='.', elinewidth=0.5, label = 'mean +/- 2*stddev')
x = np.linspace(-3,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-1.5,1.5)
axes[2].set_title('$z$')

fig.savefig('full_prob/test_train.png')

