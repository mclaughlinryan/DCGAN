# DCGAN

This program was written with PyTorch and trains a DCGAN on the CIFAR-10 dataset to achieve generation of images that are representative of the dataset. It is also trained on noised CIFAR-10 images to see if the generator recapitulates the noise in its output images. The model training for each of the scenarios was done using batches of images with 32 images per batch. The loss function used was mean squared error and the optimizer used was an Adam optimizer.

### Training on CIFAR-10 images

CIFAR-10 images:

<img width="800" alt="dcgan 4 1" src="https://github.com/mclaughlinryan/DCGAN/assets/150348966/8684d357-b810-497b-928d-5cd51135ccd2">

&nbsp;

Output images from the generator:

<img width="800" alt="dcgan 4 2" src="https://github.com/mclaughlinryan/DCGAN/assets/150348966/20ae09f1-ce38-469e-b470-460306eed835">

### Training on CIFAR-10 images with added noise

CIFAR-10 images with added Gaussian noise:

<img width="800" alt="dcgan 8 1" src="https://github.com/mclaughlinryan/DCGAN/assets/150348966/35ea7ece-5b6a-4e8b-a180-8f774343d2b3">

&nbsp;

Output images from the generator:

<img width="800" alt="dcgan 11 2" src="https://github.com/mclaughlinryan/DCGAN/assets/150348966/1c98bbed-dc45-49fa-b667-9933873b74aa">
