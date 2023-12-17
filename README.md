# DCGAN

This program trains a DCGAN on the CIFAR-10 dataset to achieve generation of images that are representative of the dataset. It is also trained on noised CIFAR-10 images to see if the generator recapitulates the noise in its output images. This program was written with PyTorch. The model training for each of the scenarios was done using batches of images with 32 images per batch. The training used a mean squared error loss function and an Adam optimizer.

#### Training on CIFAR-10 images

Images from the CIFAR-10 dataset:

![dcgan 4 1](https://github.com/mclaughlinryan/DCGAN/assets/150348966/8684d357-b810-497b-928d-5cd51135ccd2)
