import torch 
import torch.nn as nn 
import torch.nn.functional as F
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
#plt.show() 

"""Next, let's use random_split helper function to set aside 10_000 images for our validation set"""
VAL_SIZE = 10_000
TRAIN_SIZE = len(dataset) - VAL_SIZE
train_ds, val_ds = random_split(dataset, [TRAIN_SIZE, VAL_SIZE])  

"""We can now create PyTorch data loaders for the training and validation dataset"""
BATCH_SIZE = 128
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, pin_memory=True)

"""Let's visualize a batch of data in a grid using the make_grid function from torchvision. We'll use the 
.permute method on the tensor to move the color channels to the last dimension, as expected by matplotlib"""
for images, _ in train_dl: 
    #print(f"Images shape: {images.shape}")
    plt.figure(figsize=(16, 8))
    plt.axis(False) 
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break 

"""Hidden Layers, Activation Functions and Non-Linearity 
We'll create a neural network with two layers: hidden layer and output layer. Additionally, we'll use an activation function between the two layers. 
Let's look at a step-by-step example to learn how hidden layers and activation functions can help capture non-linear relationshipsss between 
inputs and outputs. 
First, let's create a batch of input tensors. We'll flatten the 1x28x28 images into vectors of size 784, so they can 
be passed into an nn.Linear object"""
for images, labels in train_dl: 
    #print(images.shape)
    inputs = images.reshape(-1, 784)
    #print(inputs.shape)  

    """Next, let's create an nn.Linear object, which will serve as our hidden layer. 
    We'll set the size of the output from the hidden layer to 32. This number can be increased or decreased to change the 
    learning capacity of the model"""
    input_size = inputs.shape[-1] # 784
    hidden_size = 32 
    layer1 = nn.Linear(input_size, hidden_size) 

    """We can now compute the intermediate outputs for the batch of images by passing 
    inputs through layer1""" 
    layer1_outputs = layer1(inputs) 
    #print(f"layer1 outputs shape: {layer1_outputs.shape}")  #[128, 32]

    """The image of size 784 are transformed into intermediate output vectors of length 32 by performing
    a matrix multiplication of inputs matrix with the transposed weights matrix of layer1 and adding the bias. We can 
    verify this using torch.allclose.""" 
    layer1_outputs_direct = inputs @ layer1.weight.t() + layer1.bias 

    #print(torch.allclose(layer1_outputs, layer1_outputs_direct)) #  True

    """Thus, layer1_outputs and inputs have a linear relationship i.e. each element of layer1_outputs is a weighted sum of 
    elements from inputs. Thus, even as we train model and modify the weights, layer1 can only capture linear relationships between inputs and outputs"""

    """Next, we'll use the Rectified Linear Unit(ReLU) function as the activation function. 
    It has the formula relu(x) = max(0, x). i.e itg simpy replaces negative values in a give tensor with the value 0. ReLU
    is a non-linear function""" 
    relu_out = F.relu(torch.tensor([[1, -1, 0], 
                               [-0.1, .2, 3]]))
    
    """Let's apply the function to layer1_outputs""" 
    relu_outputs = F.relu(layer1_outputs) 
    # print(relu_outputs.shape) # [128, 32] 
    #print(f"min layer1 outputs: {torch.min(layer1_outputs)}")
    #print(f"min relu outputs: {torch.min(relu_outputs)}")

    """Now that we've applied a non-linear activation function, relu_outputs and inputs do not have a linear relationship. 
    We refer to ReLU as the activation function, because, for each input certain outputs are activated (those with non-zero values) 
    while others are turned off (those with zero values) 
    Next, let's create an output layer to convert vectors of length hidden_size in relu_outputs into vectors of length 10, which 
    is the desired output of our model(since there are 10 target labels)"""
    output_size = 10 
    layer2 = nn.Linear(hidden_size, output_size)  
    layer2_outputs = layer2(relu_outputs)
    #print(layer2_outputs.shape) #[128, 10]  

    """As expected, layer2_outputs contains a batch of vectors of size 10. We can now use 
    this output to compute the loss using F.cross_entropy_loss and adjust the weights of layer1 and layer2"""
    loss = F.cross_entropy(layer2_outputs, labels) 
    
    """Thus, our model transforms inputs into layer2 by applying a linear transformation (using layer1), 
    followed by a non-linear actibvvation(using F.relu), followed by another linear transformation (using layer2). 
    Let's verify this by recomputing the outputs using basic matrix operations"""

    # Expanded vrsion of layer2(F.relu(layer1(inputs))) 
    outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias 

    print(torch.allclose(layer2_outputs, outputs, 1e-3))
    break 