import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torch.utils.data import random_split

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor 
from torchvision.utils import make_grid 

import numpy as np

import matplotlib.pyplot as plt 

dataset = MNIST(root="data/", 
                download=True, 
                transform=ToTensor())  
test_dataset = MNIST(root="data/", 
                     download=True, 
                     transform=ToTensor(), 
                     train=False)
"""Let's look at a couple of images from the dataset. The images are converted to PyTorch tensors with 
the shape 1x28x28 (the dimensions represent color channels, width, and height). We can use plt.imshow to display the images.
However, plt.imshow expects channels to be the last dimension in an image tensor, so we use 
the permute method to reorder the dimensions of the image"""
image, label = dataset[0] 
#print(image.shape)
plt.figure(figsize=(10, 10))
plt.imshow(image.permute(1, 2, 0), cmap="gray") 
plt.title(f"Label: {label}")

image, label = dataset[10] 
#print(image.shape)
plt.figure(figsize=(10, 10))
plt.imshow(image[0], cmap="gray") 
plt.title(f"Label: {label}")
plt.show() 

"""Next, let's use random_split helper function to set aside 10_000 images for our validation set"""
train_ds, val_ds = random_split(dataset, [50_000, 10_000]) 
print(len(train_ds))
