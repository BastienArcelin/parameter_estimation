
GalSim galaxies
- param_estimation_isolated.py : Isolated noiseless galaxy. Three options for parameter estimation (dense + latent space, no latent space, BNN RT or Flipout). 
- param_estimation_blended.py : Blended galaxies. Three options for parameter estimation (dense + latent space, no latent space, BNN)
- param_estimation_blended_psf.py : Same as above with PSF as a second input.

DC2 dataset
- param_estimation_dc2_one_input.py : parameter estimation for ellipticity only : without latent space and BNN options. 
- param_estimation_dc2.py : parameter estimation for ellipticity + redshift : lots of options
- param_estimation_dc2_vae.py : VAE for DC2 images using PSF as input.
- redshift_estimation.py : redshift only from BNN or CNN

- VAE_dc2_training_test_CC.py : a test to train a VAE, use a dense neural network to do parameter estimation from the latent space, and add a normalizing flow to match the parameter distribution as output. Probably did not work.
