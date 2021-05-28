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
from tools_for_VAE import utils, vae_functions, generator, model, boxplot, plot

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Multiply, Reshape, Flatten, BatchNormalization, Conv2D, Conv3D, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda,  concatenate
import tensorflow_probability as tfp


import tensorflow as tf

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


saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_dc2/v_vae_param'
step_size = 150


steps_per_epoch = int(100000/batch_size)
validation_steps = int(20000/batch_size)

root_dir_v1 = '/pbs/home/b/barcelin/sps_link/data/dc2_test/24.5/training/' # mag cut at 24.5 in r band
#root_dir_v2 = '/pbs/home/b/barcelin/sps_link/data/dc2_test/test_mag_26.5/' # mag cut at 26.5 in r band



tfd = tfp.distributions
tfb = tfp.bijectors
encoded_size = 32
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),reinterpreted_batch_ndims=1)

def create_model_wo_ls_peak_pooling_vae(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 

    h = BatchNormalization()(input_layer_1)
    #h = input_layer_1
    h_2 = input_layer_2
    for i in range(len(filters)):
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h_2)
        h_2 = PReLU()(h_2)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h)
        h = PReLU()(h)

        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same')(h)
        h = PReLU()(h)
        h_2 = Conv2D(filters[i], (kernels[i],kernels[i]), activation=None, padding='same', strides=(2,2))(h_2)
        h_2 = PReLU()(h_2)

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


vae = create_model_wo_ls_peak_pooling_vae(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
vae.load_weights('/sps/lsst/users/barcelin/TFP/weights/test_dc2/test_from_latent_space/cp-{epoch:04d}.ckpt')


encoder = Model([vae.layers[0].input,vae.layers[2].input], vae.layers[42].output) #vae.layers[44].output[0]



input_layer_1 = Input(shape=(input_shape)) 
input_layer_2 = Input(shape=(input_shape)) 

out = encoder([input_layer_1,input_layer_2])


tfd = tfp.distributions
tfb = tfp.bijectors
encoded_size = 32
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(32), scale=1.),reinterpreted_batch_ndims=1)


def param_estimation_net(vae, encoder, final_dim):
    input_layer_1 = Input(shape=(input_shape)) 
    input_layer_2 = Input(shape=(input_shape)) 
    
    encoder.trainable = False
    h = encoder([input_layer_1,input_layer_2])
    
    #h = tfp.layers.MultivariateNormalTriL(32,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01))(tf.cast(h,tf.float32))

    h = Dense(256, activation = 'tanh')(h)
    h = Dense(256, activation = 'tanh')(h)
    h = Dense(256, activation = None)(h)
    h = PReLU()(h)
    h = Dense(128, activation = None)(h)
    h = PReLU()(h)
    
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)
    
    return Model((input_layer_1, input_layer_2), h)


param_estim_net = param_estimation_net(vae, encoder, final_dim)
param_estim_net.summary()



def kl_metric(y_true, y_pred):
    return K.sum(param_estim_net.losses)

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
mse = tf.keras.losses.MeanSquaredError()

param_estim_net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
              loss=negative_log_likelihood,
              metrics = ['mse', 'acc'],#,kl_metric],
              experimental_run_tf_function=False,
              )


import tensorflow
## Compute lensed ellipticities from shear and 
def calc_lensed_ellipticity_1(es1, es2, gamma1, gamma2, kappa):
    gamma = gamma1 + gamma2*1j # shear (as a complex number)
    es =  es1 + es2*1j # intrinsic ellipticity (as a complex number)
    g = gamma / (1.0 - kappa) # reduced shear
    e = (es + g) / (1.0 + g.conjugate()*es) # lensed ellipticity
    return np.real(e)

def calc_lensed_ellipticity_2(es1, es2, gamma1, gamma2, kappa):
    gamma = gamma1 + gamma2*1j # shear (as a complex number)
    es =   es1 + es2*1j # intrinsic ellipticity (as a complex number)
    g = gamma / (1.0 - kappa) # reduced shear
    e = (es + g) / (1.0 + g.conjugate()*es) # lensed ellipticity
    return np.imag(e)

def calc_lensed_ellipticity(es1, es2, gamma1, gamma2, kappa):
    gamma = gamma1 + gamma2*1j # shear (as a complex number)
    es =   es1 + es2*1j # intrinsic ellipticity (as a complex number)
    g = gamma / (1.0 - kappa) # reduced shear
    e = (es + g) / (1.0 + g.conjugate()*es) # lensed ellipticity
    return np.absolute(e)

class BatchGenerator_dc2_deconv_noisy_2(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e, net, saving_path, prop = 0, step_size = 10000):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        self.path = path
        
        self.epoch = 1
        self.prop = prop
        self.do_norm = do_norm
        self.denorm = denorm
        self.net = net
        self.saving_path = saving_path
        self.step_size = step_size

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.epoch +=1 
        #print(self.epoch)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # Change the proportion of noisy data every step_size epochs:
        if self.epoch == self.step_size:
            self.prop +=1
            self.epoch = 0
            print(self.prop)
            
        if self.prop == 11:
            self.prop=10

        if self.trainval_or_test == 'training':
            data_path = os.path.join(self.path,'training/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]
        
        if self.trainval_or_test == 'validation':
            data_path = os.path.join(self.path,'validation/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]

        if self.trainval_or_test == 'test':
            data_path = os.path.join(self.path,'test/')
            list_of_samples_noiseless = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_noiseless_sample_')]
            list_of_samples_noisy = [x for x in utils.listdir_fullpath(data_path) if x.startswith(data_path+'img_cropped_sample_')]


        list_of_samples_noiseless_chosen = np.random.choice(list_of_samples_noiseless, size = 10-self.prop)
        list_of_samples_noisy_chosen = np.random.choice(list_of_samples_noisy, size = self.prop)

        list_of_samples_used = [*list_of_samples_noiseless_chosen, *list_of_samples_noisy_chosen]

        sample_filename = np.random.choice(list_of_samples_used, size = 1)[0]#, p=self.p)
        sample = np.load(sample_filename, mmap_mode = 'c')
        #print(sample_filename)
        
        if sample_filename.startswith(data_path+'img_cropped_sample_')==True:
            #print('first')
            data = pd.read_csv(sample_filename.replace('img_cropped_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_cropped_sample','psf_cropped_sample'), mmap_mode = 'c')
        else:
            #print('second')
            data = pd.read_csv(sample_filename.replace('img_noiseless_sample','img_noiseless_data').replace('.npy','.csv'))
            psf = np.load(sample_filename.replace('img_noiseless_sample','psf_cropped_sample'), mmap_mode = 'c')
           
        data['weights']=np.sqrt(np.abs(data['e1'])+np.abs(data['e2']))
        #print(data['weights'])
        #print(np.min(data['weights']), np.max(data['weights']))

        ell_1 = np.array(data['e1'])
        ell_2 = np.array(data['e2'])
        shear_1 = np.array(data['shear_1'])
        shear_2 = np.array(data['shear_2'])
        convergence = np.array(data['convergence'])
        
        ellipticity = calc_lensed_ellipticity(-ell_1, ell_2, shear_1, shear_2, convergence)
        ellipticity_conversion = lambda e: 2*e / (1.0+ellipticity[:len(e)]*ellipticity[:len(e)])

        ellipticity_1 = ellipticity_conversion(calc_lensed_ellipticity_1(-ell_1, ell_2, shear_1, shear_2, convergence))
        ellipticity_2 = ellipticity_conversion(calc_lensed_ellipticity_2(-ell_1, ell_2, shear_1, shear_2, convergence))

        data['ellipticity_1_lensed'] = ellipticity_1
        data['ellipticity_2_lensed'] = ellipticity_2

        new_data = data#[(np.abs(data['ellipticity_1_lensed'])<=1.) &
        #                (np.abs(data['ellipticity_2_lensed'])<=1.)]# 
                        #(data['snr_r']>20)]#snr_r
                        #(np.abs(data['blendedness'])==0.)]# &
        
        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = new_data['weights']/np.sum(new_data['weights']))
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))

        self.produced_samples += len(indices)
        
        x_1 = np.tanh(np.arcsinh(sample[indices][:,:,:,self.bands]))
        x_2 = psf[indices][:,:,:,self.bands]
        y = np.zeros((self.batch_size, 2))
        ellipticity_1 = new_data['ellipticity_1_lensed'][indices]
        ellipticity_2 = new_data['ellipticity_2_lensed'][indices]

        #flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            x_1 = np.flip(x_1, axis=2)
            x_2 = np.flip(x_2, axis=2)
            y[:,0] = -ellipticity_1
            y[:,1] = -ellipticity_2
        elif rand == 2:
            x_1 = np.swapaxes(x_1, 2, 1)
            x_2 = np.swapaxes(x_2, 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = ellipticity_2
        elif rand == 3:
            x_1 = np.swapaxes(np.flip(x_1, axis=2), 2, 1)
            x_2 = np.swapaxes(np.flip(x_2, axis=2), 2, 1)
            y[:,0] = ellipticity_1
            y[:,1] = -ellipticity_2
        else:
            y[:,0] = -ellipticity_1
            y[:,1] = ellipticity_2
        if len(self.bands)==1:
            x_1 = np.expand_dims(x_1, axis=-1)
            x_2 = np.expand_dims(x_2, axis=-1)

        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return (x_1, x_2), y
        elif self.trainval_or_test == 'test':
            return (x_1, x_2), y



images_dir = '/pbs/home/b/barcelin/sps_link/data/dc2_test/24.5/'
list_of_samples = [[x for x in utils.listdir_fullpath(os.path.join(images_dir,'training/')) if x.startswith(os.path.join(images_dir,'training/')+'img_cropped_sample')][0]]

training_generator = BatchGenerator_dc2_deconv_noisy_2(bands,
                                    images_dir,
                                    list_of_samples, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None,
                                    net = param_estim_net,
                                    saving_path = saving_path,
                                    prop = 0,
                                    step_size = step_size)

list_of_samples = [[x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation/')) if x.startswith(os.path.join(images_dir,'validation/')+'img_cropped_sample')][0]]

validation_generator = BatchGenerator_dc2_deconv_noisy_2(bands,
                                    images_dir,
                                    list_of_samples, 
                                    total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None,
                                    net = param_estim_net,
                                    saving_path = saving_path,
                                    prop = 0,
                                    step_size = step_size)


# Callbacks
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)#mse en TF2
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
#checkpointer_acc = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'/acc/weights_noisy_v4.{epoch:02d}-{val_acc:.2f}.ckpt', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max', period=1)

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
        #print(self.step_siz)
        print(self.save_path)
        if (self.epoch == self.step_siz):
            self.epoch =0
            self.network.save_weights(self.save_path+'cp-'+str(self.epoch)+'.ckpt')
        self.epoch +=1
        #print(self.epoch)

cb = save_model_step(param_estim_net, saving_path, step_size)

callbacks = [checkpointer_mse, checkpointer_loss, cb]#, checkpointer_acc]#, alpha_changer]#, alpha_changer]#, WandbCallback()]#, alpha_changer]



######## Train the network
hist = param_estim_net.fit(training_generator, epochs=2000,
                    steps_per_epoch=steps_per_epoch,
                    verbose=2,
                    shuffle=True,
                    validation_data=validation_generator,
                    validation_steps=validation_steps,
                    callbacks=callbacks)



test = training_generator.__getitem__(3)

training_data = test[0]#[0], test[0][1]]
training_labels = test[1]

