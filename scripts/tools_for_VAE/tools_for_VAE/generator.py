# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import tensorflow.keras
import pandas as pd
import scipy
from scipy.stats import norm

from random import choice

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils
import tensorflow as tf
import galsim

######### FOR PSF GENERATION 


fwhm_lsst = 0.65 ## Fixed at median value : Fig 1 : https://arxiv.org/pdf/0805.2366.pdf

PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)
pixel_scale_lsst = 0.2
img_size = 64
#################### FILTERS ###################
filters = {}
euclid_filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/EUCLID_Filters/')
lsst_filters_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/share_galsim/bandpasses')

# read in the Euclid NIR filters
filter_names_euclid_nir = 'HJY'
filter_names_euclid_vis = 'V'

for filter_name in filter_names_euclid_nir:
    filter_filename = os.path.join(euclid_filters_dir, 'Euclid_NISP0.{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

filter_filename = os.path.join(euclid_filters_dir, 'Euclid_VIS.dat')
filters['V'] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
filters['V'] = filters[filter_name].thin(rel_err=1e-4)

# read in the LSST filters
filter_names_lsst = 'ugrizy'
for filter_name in filter_names_lsst:
    filter_filename = os.path.join(lsst_filters_dir, 'LSST_{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

filter_names_all = 'HJYVugrizy'


####################


class BatchGenerator(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
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
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

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
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        #sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        sample = np.load(sample_filename, mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('images.npy','data.csv'))

        new_data = data[(np.abs(data['e1'])<=1.) &
                        (np.abs(data['e2'])<=1) ]

        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))
            #print(indices)
        self.produced_samples += len(indices)

        x = sample[indices,1][:,self.bands]
        #print(x.shape)
        
        y = np.zeros((self.batch_size, 3))
        y[:,0] = np.array(new_data['e1'][indices])#np.exp(np.array(new_data['e1'][indices]))*2 
        y[:,1] = np.array(new_data['e2'][indices])#np.exp(np.array(new_data['e2'][indices]))*2
        y[:,2] = np.array(new_data['redshift'][indices])
        
        # Preprocessing of the data to be easier for the network to learn
        if self.do_norm:
            x = utils.norm(x, self.bands, n_years = 5)
        if self.denorm:
            x = utils.denorm(x, self.bands, n_years = 5)
        
        x = np.transpose(x, axes = (0,2,3,1))
        
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return x, y
        elif self.trainval_or_test == 'test':
            return x, y#, data.loc[indices], indices



class BatchGenerator_random_coord_psf(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands,path, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
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
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

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
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        #sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        sample = np.load(sample_filename, mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('images.npy','data.csv'))
        shifts = np.load(self.path+self.trainval_or_test+'/shifts/'+sample_filename[-38:].replace('images.npy','shifts.npy'))

        #data = data.replace(to_replace = 10.,value = 0)
        #print(data)
        new_data = data[(np.abs(data['e1_fit_0'])<=1.) &#e1_0
                        (np.abs(data['e2_fit_0'])<=1.) &#e2_0
                        (np.abs(data['e1_fit_1'])<=1.) &#e1_1
                        (np.abs(data['e2_fit_1'])<=1.) &#e2_1
                        (np.abs(data['e1_fit_2'])<=1.) &#e1_2
                        (np.abs(data['e2_fit_2'])<=1.) &#e2_2
                        (np.abs(data['e1_fit_3'])<=1.) &#e1_3
                        (np.abs(data['e2_fit_3'])<=1.) ]#e2_3
        #print(new_data['nb_blended_gal'])

        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))
            #print(indices)
        self.produced_samples += len(indices)

        y = np.zeros((self.batch_size, 2))

        x_1 = sample[indices,-1][:,self.bands]

        x_2 = np.zeros((self.batch_size,64,64,6))

        for i in range (self.batch_size):
            z = np.random.random_integers(new_data['nb_blended_gal'][indices[i]])
            fwhm_lsst = new_data['fwhm_lsst'][indices[i]]
            PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)
            psf = PSF_lsst.shift((shifts[indices,z-1][i,0],shifts[indices,z-1][i,1]))
            temp_img = galsim.ImageF(img_size, img_size, scale=pixel_scale_lsst)
            psf.drawImage(image=temp_img)
            for m in range(6):
                x_2[i,:,:,m]=temp_img.array.data
            y[i,0] = np.array(new_data['e1_fit_'+str(z-1)][indices[i]])
            y[i,1] = np.array(new_data['e2_fit_'+str(z-1)][indices[i]])
            #y[i,2] = np.log(np.array(new_data['redshift_'+str(z-1)][indices[i]]))
        
        # Preprocessing of the data to be easier for the network to learn
        if self.do_norm:
            x_1 = utils.norm(x_1, self.bands, n_years = 5)
        if self.denorm:
            x_1 = utils.denorm(x_1, self.bands, n_years = 5)
        
        x_1 = np.transpose(x_1, axes = (0,2,3,1))
        
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return (x_1, x_2), y#[tf.cast(x_1, tf.float32), tf.cast(x_2, tf.float32)], tf.cast(y, tf.float32)
        elif self.trainval_or_test == 'test':
            return (x_1, x_2), y

