import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim

import random
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils

import torchvision.datasets
import torchvision.transforms as transforms

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

set_fraction = (3200 / len(dataset))  # get fraction of data to achieve dataset of 3200 images, which is 100 batches of 32 images/batch
dataset, testset, _ = torch.utils.data.dataset.random_split(dataset, [int(set_fraction * len(dataset)),
                                                                      int((1 / 2) * set_fraction * len(dataset)),
                                                                      int((1 - (3 / 2) * set_fraction) * len(dataset))],
                                                            generator=torch.Generator().manual_seed(42))

# Partial data to train GAN on for aim of project
class transformGaussianNoise():
    def __init__(self, mean=0, std=1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Addition of Gaussian noise to image set
data_noise = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                              transformGaussianNoise(0, 0.2)
                                          ]))

data_noise, testset_noise, _ = torch.utils.data.dataset.random_split(data_noise, [int(set_fraction * len(data_noise)),
                                                                                  int((1 / 2) * set_fraction * len(
                                                                                      data_noise)), int((1 - (
                3 / 2) * set_fraction) * len(data_noise))], generator=torch.Generator().manual_seed(42))

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 32

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps for generator
ngf = 32

# Size of feature maps for discriminator
ndf = 32

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Declare dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
dataloader_imperfect = torch.utils.data.DataLoader(data_noise, batch_size=batch_size,
                                                   shuffle=False, num_workers=workers)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
testloader_imperfect = torch.utils.data.DataLoader(testset_noise, batch_size=batch_size,
                                                   shuffle=False, num_workers=workers)

# Decide the device to run program on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot grid of training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Training Images (Original dataset)")
plt.imshow(
    np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(), (1, 2, 0)))

# Plot grid of noised data
imperfect_batch = next(iter(dataloader_imperfect))
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Training Images (Noised data)")
plt.imshow(np.transpose(vutils.make_grid(imperfect_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),
                        (1, 2, 0)))

# custom weights initialization for generator and discriminator network
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# DCGAN generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.02
netG.apply(weights_init)

# Print the model
print(netG)

# DCGAN discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.2
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize loss function, using MSE
criterion = nn.MSELoss()

# Create batch of latent vectors that will be used to visualize
# progression of the generator
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
# Lists to keep track of progress
img_list = []
img_list_real = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Train with real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save Losses to plot later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                img_list.append(vutils.make_grid(fake.detach(), padding=2, normalize=True))
                img_list_real.append(vutils.make_grid(real_cpu, padding=2, normalize=True))

        iters += 1

    if epoch % 10 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot of MSE loss from networks over the course of training
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot real images
plt.figure(figsize=(32, 32))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[-1].detach().cpu(), (1, 2, 0)))

# Plot fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Testing Loop on ground truth data
# Optimizers not utilized in testing phases as just want to assess performance and do not want to further train the network
# Lists to keep track of generated output and loss values during testing phases
img_list_test = [[]] * 4
img_list_real_test = [[]] * 4
G_losses_test = [[]] * 4
D_losses_test = [[]] * 4
iters = 0

print("Starting Testing Loop...")
# For each epoch
for epoch in range(1):
    # For each batch in the dataloader
    for i, data in enumerate(testloader, 0):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Test with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on real batch
        errD_real = criterion(output, label)
        D_x = output.mean().item()

        # Test with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on fake batch
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # Update G network: maximize log(D(G(z)))
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()

        # Save Losses to plot later
        G_losses_test[0].append(errG.item())
        D_losses_test[0].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(testloader) - 1)):
            with torch.no_grad():
                img_list_test[0].append(vutils.make_grid(fake, padding=2, normalize=True))
                img_list_real_test[0].append(vutils.make_grid(real_cpu, padding=2, normalize=True))

        iters += 1

    if epoch % 10 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Testing Loop on noised data
iters = 0

print("Starting Testing Loop...")
# For each epoch
for epoch in range(1):
    # For each batch in the dataloader
    for i, data in enumerate(testloader_imperfect, 0):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Test with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on real batch
        errD_real = criterion(output, label)
        D_x = output.mean().item()

        # Test with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the fake batch
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # Update G network: maximize log(D(G(z)))
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()

        # Save Losses to plot later
        G_losses_test[1].append(errG.item())
        D_losses_test[1].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(testloader_imperfect) - 1)):
            with torch.no_grad():
                img_list_test[1].append(vutils.make_grid(fake, padding=2, normalize=True))
                img_list_real_test[1].append(vutils.make_grid(real_cpu, padding=2, normalize=True))

        iters += 1

    if epoch % 10 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot of MSE loss from networks over course of testing
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Testing on Dataset Images")
plt.plot(G_losses_test[0], label="G")
plt.plot(D_losses_test[0], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Testing on Noised Dataset Images")
plt.plot(G_losses_test[1], label="G")
plt.plot(D_losses_test[1], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot real images
plt.figure(figsize=(32, 32))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real_test[0][-1].detach().cpu(), (1, 2, 0)))

# Plot fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images (Test phase / Ground truth trained)")
plt.imshow(np.transpose(img_list_test[0][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Plot noised images
plt.figure(figsize=(32, 32))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Noised images")
plt.imshow(np.transpose(img_list_real_test[1][-1].detach().cpu(), (1, 2, 0)))

# Plot fake noised images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Noised Images (Test phase / Ground truth trained)")
plt.imshow(np.transpose(img_list_test[1][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Create the generator
netG2 = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG2 = nn.DataParallel(netG2, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.02
netG2.apply(weights_init)

# Print the model
print(netG2)

# Create the discriminator
netD2 = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD2 = nn.DataParallel(netD2, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.2
netD2.apply(weights_init)

# Print the model
print(netD2)

# Setup Adam optimizers for G and D
optimizerD = optim.Adam(netD2.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG2.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
# Lists to keep track of progress
img_list = []
img_list_real = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader_imperfect, 0):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Train with real batch
        netD2.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD2(real_cpu).view(-1)
        # Calculate loss on real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG2(noise)
        label.fill_(fake_label)
        # Classify fake batch with D
        output = netD2(fake.detach()).view(-1)
        # Calculate D's loss on fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # Update G network: maximize log(D(G(z)))
        netG2.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of fake batch through D
        output = netD2(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save Losses to plot later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader_imperfect) - 1)):
            with torch.no_grad():
                img_list.append(vutils.make_grid(fake.detach(), padding=2, normalize=True))
                img_list_real.append(vutils.make_grid(real_cpu, padding=2, normalize=True))

        iters += 1

    if epoch % 10 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot of MSE loss from networks over the course of training
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot real images
plt.figure(figsize=(32, 32))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Noised Images")
plt.imshow(np.transpose(img_list_real[-1].detach().cpu(), (1, 2, 0)))

# Plot fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Noised Images")
plt.imshow(np.transpose(img_list[-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Testing Loop on noised data
iters = 0

print("Starting Testing Loop...")
# For each epoch
for epoch in range(1):
    # For each batch in the dataloader
    for i, data in enumerate(testloader_imperfect, 0):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Test with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD2(real_cpu).view(-1)
        # Calculate loss on real batch
        errD_real = criterion(output, label)
        D_x = output.mean().item()

        # Test with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG2(noise)
        label.fill_(fake_label)
        # Classify fake batch with D
        output = netD2(fake.detach()).view(-1)
        # Calculate D's loss on fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # Update G network: maximize log(D(G(z)))
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of fake batch through D
        output = netD2(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        D_G_z2 = output.mean().item()

        # Save Losses to plot later
        G_losses_test[2].append(errG.item())
        D_losses_test[2].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(testloader_imperfect) - 1)):
            with torch.no_grad():
                img_list_test[2].append(vutils.make_grid(fake, padding=2, normalize=True))
                img_list_real_test[2].append(vutils.make_grid(real_cpu, padding=2, normalize=True))

        iters += 1

    if epoch % 10 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Testing Loop on ground truth data
iters = 0

print("Starting Testing Loop...")
# For each epoch
for epoch in range(1):
    # For each batch in the dataloader
    for i, data in enumerate(testloader, 0):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Test with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD2(real_cpu).view(-1)
        # Calculate loss on real batch
        errD_real = criterion(output, label)
        D_x = output.mean().item()

        # Test with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG2(noise)
        label.fill_(fake_label)
        # Classify fake batch with D
        output = netD2(fake.detach()).view(-1)
        # Calculate D's loss on the fake batch
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # Update G network: maximize log(D(G(z)))
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD2(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()

        # Save Losses to plot later
        G_losses_test[3].append(errG.item())
        D_losses_test[3].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(testloader) - 1)):
            with torch.no_grad():
                img_list_test[3].append(vutils.make_grid(fake, padding=2, normalize=True))
                img_list_real_test[3].append(vutils.make_grid(real_cpu, padding=2, normalize=True))

        iters += 1

    if epoch % 10 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot of MSE loss from networks over the course noised set testing
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Testing on Noised Dataset Images")
plt.plot(G_losses_test[3], label="G")
plt.plot(D_losses_test[3], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Testing on Dataset Images")
plt.plot(G_losses_test[3], label="G")
plt.plot(D_losses_test[3], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot noised images
plt.figure(figsize=(32, 32))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Noised images")
plt.imshow(np.transpose(img_list_real_test[2][-1].detach().cpu(), (1, 2, 0)))

# Plot fake noised images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Noised Images (Test phase / Noise trained)")
plt.imshow(np.transpose(img_list_test[2][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Plot real images
plt.figure(figsize=(32, 32))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real_test[3][-1].detach().cpu(), (1, 2, 0)))

# Plot fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images (Test phase / Noise trained)")
plt.imshow(np.transpose(img_list_test[3][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Overall comparison of GAN testing after training with complete samples vs. partial/noisy samples
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[-1].detach().cpu(), (1, 2, 0)))

# Plot fake images from the last epoch
plt.figure(figsize=(10, 5))
plt.axis("off")
plt.title("Fake Images (Ground truth training)")
plt.imshow(np.transpose(img_list_test[0][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Plot fake images from the last epoch
plt.figure(figsize=(10, 5))
plt.axis("off")
plt.title("Fake Images (Noised data training)")
plt.imshow(np.transpose(img_list_test[3][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Plot of MSE loss from networks over the course noised set testing
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss (Ground truth training) During Testing on Dataset Images")
plt.plot(G_losses_test[0], label="G")
plt.plot(D_losses_test[0], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss (Noised data training) During Testing on Dataset Images")
plt.plot(G_losses_test[3], label="G")
plt.plot(D_losses_test[3], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Noised Images")
plt.imshow(np.transpose(img_list[-1].detach().cpu(), (1, 2, 0)))

# Plot fake images from the last epoch
plt.figure(figsize=(10, 5))
plt.axis("off")
plt.title("Fake Noised Images (Ground truth training)")
plt.imshow(np.transpose(img_list_test[1][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Plot fake images from the last epoch
plt.figure(figsize=(10, 5))
plt.axis("off")
plt.title("Fake Noised Images (Noised data training)")
plt.imshow(np.transpose(img_list_test[2][-1].detach().cpu(), (1, 2, 0)))
plt.show()

# Plot of MSE loss from networks over the course noised set testing
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss (Ground truth training) During Testing on Noised Dataset Images")
plt.plot(G_losses_test[1], label="G")
plt.plot(D_losses_test[1], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss (Noised data training) During Testing on Noised Dataset Images")
plt.plot(G_losses_test[2], label="G")
plt.plot(D_losses_test[2], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
