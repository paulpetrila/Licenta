# original implementation was in Theano and Pylearn2, so I'm rewritting it in Pytorch, because that's what I'm going to use
# source : https://github.com/goodfeli/adversarial

# system dependencies
import argparse
import os
import numpy as np
import math

# vision
import torchvision.transforms as transforms
from torchvision.utils import save_image

# working with datasets 
from torch.utils.data import DataLoader
from torchvision import datasets

# neural nets dependencies
import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser(prog="gan", 
                                 description="A Pytorch implementation of the original Goodfellow GAN",
                                 epilog="The defaults are good enough for most types of images, training it will take a while")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs while training")
parser.add_argument("--batch_size", type=int, default=64, help="size of each batch")

# Adam optimizer: https://arxiv.org/pdf/1412.6980
parser.add_argument("--learning_rate",type=float, defualt=0.0002, help = "adam: learning rate")
parser.add_argument("--b1", type=float,default=0.5,help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float,default=0.999,help="adam: decay of first order")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during the batch generation")

parser.add_argument("--latent_dim", type=int,default=100, help="dimensionality of latent space")

parser.add_argument("--img_size", type=int, default=28,help="size of each image dimension")

parser.add_argument("--chanels", type=int, default=1, help="number of image channels, defaults to 1(grayscale)")

parser.add_argument("--sample_interval", type=int,default=400,help="interval between image samples")

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

print(f"{"Using" if cuda else "Not using" } cuda")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
       
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )  
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()



