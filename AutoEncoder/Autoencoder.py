########## Autoencoder #############


#################### Import ####################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import argparse
parser = argparse.ArgumentParser('Autoencoder training and evaluation script')
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--latent_dims', default=10, type=int, help='latent dimensions')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--use_gpu', action='store_true', help='use GPU')
parser.add_argument('--pretrained', action='store_true', help='use pretrained weights')


args = parser.parse_args()


#################### Parameter Settings ####################
num_workers = args.num_workers
latent_dims = args.latent_dims
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
use_gpu = args.use_gpu
pretrained = args.pretrained

print("#########################")
print("# latent_dims :", latent_dims)
print("# num_epochs :", num_epochs)
print("# batch_size :", batch_size)
print("# learning_rate :", learning_rate)
print("# num_workers :", num_workers)
print("# use gpu :", use_gpu)
print("# use pretrained :", pretrained)
print("#########################")

#################### MNIST Data Loading ####################
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#################### Autoencoder Definition ####################
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = 64
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64 * 2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv1(x)) # last layer before output is tanh, since the images are normalized and 0-centered
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon
    
autoencoder = Autoencoder()
print(autoencoder)

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)

num_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

#################### Pretrained Weights ###################
if pretrained == True:
    model_state_dict = torch.load("./my_autoencoder_100epoch.pth", map_location=device)
    autoencoder.load_state_dict(model_state_dict)


#################### Train Autoencoder ####################

elif pretrained == False:
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    # set to training mode
    autoencoder.train()

    train_loss_avg = []

    print('Training ...')
    for epoch in range(num_epochs):
        train_loss_avg.append(0)
        num_batches = 0
        
        for image_batch, _ in train_dataloader:
            
            image_batch = image_batch.to(device)
            
            # autoencoder reconstruction
            image_batch_recon = autoencoder(image_batch)
            
            # reconstruction error
            loss = F.mse_loss(image_batch_recon, image_batch)
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()
            
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch, num_epochs, train_loss_avg[-1]))

    torch.save(autoencoder.state_dict(), './my_autoencoder_{}epoch.pth'.format(str(num_epochs+1)))
#################### Plot Training Curve ####################
    plt.ion()

    fig = plt.figure()
    plt.plot(train_loss_avg)
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction error')
    plt.savefig("./1.Plot Training Curve.png")
    plt.show()
#################### Visualize Reconstructions ####################

plt.ion()

autoencoder.eval()

# This function takes as an input the images to reconstruct
# and the name of the model with which the reconstructions
# are performed
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def visualise_output(images, model):

    with torch.no_grad():

        images = images.to(device)
        images = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.savefig("./3.Autoencoder reconstruction.png")
        plt.show()

images, labels = next(iter(test_dataloader))

# First visualise the original images
# Original images
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.savefig("./2.Original images.png")
plt.show()

# Reconstruct and visualise the images using the autoencoder
# Autoencoder reconstruction
visualise_output(images, autoencoder)

#################### Interpolate in Latent Space ####################
autoencoder.eval()

def interpolation(lambda1, model, img1, img2):
    
    with torch.no_grad():

        # latent vector of first image
        img1 = img1.to(device)
        latent_1 = model.encoder(img1)

        # latent vector of second image
        img2 = img2.to(device)
        latent_2 = model.encoder(img2)

        # interpolation of the two latent vectors
        inter_latent = lambda1 * latent_1 + (1- lambda1) * latent_2

        # reconstruct interpolated image
        inter_image = model.decoder(inter_latent)
        inter_image = inter_image.cpu()
    
    return inter_image
    
# sort part of test set by digit
digits = [[] for _ in range(10)]
for img_batch, label_batch in test_dataloader:
    for i in range(img_batch.size(0)):
        digits[label_batch[i]].append(img_batch[i:i+1])
    if sum(len(d) for d in digits) >= 1000:
        break;

# interpolation lambdas
lambda_range=np.linspace(0,1,10)
# print(lamb)
fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for ind,l in enumerate(lambda_range):
    inter_image=interpolation(float(l), autoencoder, digits[7][0], digits[1][0])
   
    inter_image = to_img(inter_image)
    
    image = inter_image.numpy()
   
    axs[ind].imshow(image[0,0,:,:], cmap='gray')
    axs[ind].set_title('lambda_val='+str(round(l,1)))
plt.savefig("./4.Interpolate in Latent Space.png")
plt.show()

#################### Random Latent Vector (Autoencoder as Generator) ####################
autoencoder.eval()

with torch.no_grad():
    # approx. fit a multivariate Normal distribution (with diagonal cov.) to the latent vectors of a random part of the test set
    images, labels = next(iter(test_dataloader))
    images = images.to(device)
    latent = autoencoder.encoder(images)
    latent = latent.cpu()

    mean = latent.mean(dim=0)
    std = (latent - mean).pow(2).mean(dim=0).sqrt()

    # sample latent vectors from the normal distribution
    latent = torch.randn(128, latent_dims)*std + mean

    # reconstruct images from the latent vectors
    latent = latent.to(device)
    img_recon = autoencoder.decoder(latent)
    img_recon = img_recon.cpu()

    fig, ax = plt.subplots(figsize=(5, 5))
    show_image(torchvision.utils.make_grid(img_recon[:100],10,5))
    plt.savefig("./5.Random Latent Vector(Autoencoder as Generator).png")
    plt.show()

print("##### Done #####")