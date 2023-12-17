# DCGAN

This program trains a DCGAN on the CIFAR-10 dataset to achieve generation of images that are representative of the dataset. It is also trained on noised CIFAR-10 images to see if the generator recapitulates the noise in its output images. This program was written with PyTorch. The model training for each of the scenarios was done using 1,000 batches of images with 32 images in each batch. The network was trained for 10 epochs and used a mean squared error loss function and an Adam optimizer.
