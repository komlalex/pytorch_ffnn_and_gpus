import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torch.utils.data import random_split 
from torchinfo import summary


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

    # Expanded version of layer2(F.relu(layer1(inputs))) 
    outputs = (F.relu(inputs @ layer1.weight.t() + layer1.bias)) @ layer2.weight.t() + layer2.bias 

    # print(torch.allclose(layer2_outputs, outputs, 1e-3)) True 

    """If we had not included a non-linear activation between the two linear layers, the final relationship 
    between inputs and outputs would be linear. A simple refactoring of computations illustrates this"""
    # Same as layer2(layer1(inputs))  
    outputs2 = (inputs @ layer1.weight.t() + layer1.bias) @ layer2.weight.t() + layer2.bias 

    # Create a single layer to replace the two linear layers 
    combined_layer = nn.Linear(input_size, output_size)
    combined_layer.weight.data = layer2.weight @ layer1.weight 
    combined_layer.bias.data = layer1.bias @ layer2.weight.t() + layer2.bias

    # Same as combined_layer(inputs) 
    outputs3 = inputs @ combined_layer.weight.t() + combined_layer.bias 
    #print(torch.allclose(outputs2, outputs3, 1e-3))
    break  

"""Let's define the accuracy function"""
def accuracy(outputs, y_true): 
    """Compute the model's accuracy. It is used in validation step"""
    y_preds = torch.argmax(outputs, dim=1) 
    return torch.tensor(torch.sum(y_preds == y_true).item() / len(y_true))
"""Model
We are now ready to define our model. As discussed above, we'll create a neural network with 
one hidden laye. Here's what that means:
* Instead of using a single nn.Linear object to transform a batch of inputs (pixel intensities)
into outputs (class probabilities). We'lluse two nn.Linear objects. Each of these is 
called a layer in the network. 
* The first layers (also know as the hidden layer) will transform the input matrix of shape batch_size x 784
into an intermediate output matrix of shape batch_size x hidden_size. The parameter hidden_size
can be configured manually (e.g 32 or 64). 
* We'll then apply a non-linear activation function to the intermediate outputs. The activation 
function transforms the individual elements in the matrix. 
* The result of the activation function, which is also of the size batch_size x hidden_size, is passed into 
the second layer (also known as the output layer). The second layer transforms it into a matrix of size
batch_size x 10. We can use this output to compute the loss and adjust weights using gradient descent. 

Let's define the model by extending the nn.Module class of PyTOrch
"""
class MnistModel(nn.Module): 
    """Feedforward neural network with 1 hidden layer""" 
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__() 
        # hidden layer 
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer 
        self.linear2 = nn.Linear(hidden_size, out_size) 

    def forward(self, xb) -> torch.Tensor: 
        # Flatten the image tensors 
        xb = xb.view(xb.size(0), -1)  # Flatten tensors
        # Get intermediate outputs using hidden layer 
        out = self.linear1(xb)
        # Apply activation functio 
        out = F.relu(out) 
        # Get predictions using output layer 
        out = self.linear2(out)  
        return out

    def training_step(self, batch): 
        images, labels = batch 
        out = self(images)                 # Generates predictions
        loss = F.cross_entropy(out, labels) # Calcualate loss
        return loss 
    
    def validation_step(self, batch): 
        images, labels = batch 
        out = self(images)                     # Generate predictions
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels) 
        return {"val_loss": loss, "val_acc": acc}  
    
    def validation_epoch_end(self, outputs): 
        batch_losses = [x["val_loss"] for x in outputs] 
        epoch_loss = torch.stack(batch_losses).mean() # Combnine losses
        batch_accs= [x['val_acc'] for x in outputs]  
        epoch_acc = torch.stack(batch_accs).mean() # Combine accuracies 
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()} 
    
    def epoch_end(self, epoch, result): 
        print(f"\33[32m Epoch: {epoch+1} | val_loss: {result["val_loss"]: .4f} | val_acc: {result["val_acc"]: .4f}")
INPUT_SIZE = 784 
HIDDEN_SIZE = 32 
NUM_CLASSES = len(dataset.classes) 

model = MnistModel(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES) 

"""Let's look at a summary of our model"""
#print(summary(model))

"""Let's look at the model's parameters. We'll expect to see one weight and bias matrix for each of the 
layers"""
for t in model.parameters(): 
    #print(t.shape)
    pass 

"""Let's try and generate some inputs using our model. We'll take the first batch of 128 images 
from our dataset and pass them into the model"""

for images, labels in train_dl: 
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels) 
    #print(f"Loss: {loss.item()}")  
    #print(f"Output shape: {outputs.shape}")
    #print(f"Sample outputs: \n {outputs[:2].data}")
    break 

"""Using GPUs
As the sizes of our model and datasets increases, we need to use GPUs to train our models with 
a measurable amount of time. GPUs contain hundreds of cores optimized for performing expensive 
matrix operations on floating-point numbers quickly, making them ideal for training deep neural 
networks.

We can check if a GPU is available and the required NVIDIA CUDA drivers are installed using 
torch.cuda.is_available 
"""

#print(torch.cuda.is_available()) 

"""Let's define a helper function to ensure that our code uses the GPU if it is available 
and defaults to using the CPU  if it isn't"""

def get_default_device(): 
    """Pick GPU if available, else CPU""" 
    if torch.cuda.is_available():
        return torch.device("cuda")
    else: 
        return torch.device("cpu") 

device = get_default_device() 

"""Next, let's define a function that can move data and model to a chosen device"""

def to_device(data, device): 
    """Move tensor(s) to a chosen device""" 
    if isinstance(data, (list, tuple)): 
        return [to_device(x, device) for x in data] 
    return data.to(device, non_blocking=True) 

for images, labels in train_dl: 
    #print(images.shape) 
    images = to_device(images, device) 
    #print(images.device) 
    break 

"""Finally, we define a DeviceDataLoader class to wrp our existing data loaders and move batches of data 
to the selected device. Interestingly, we don't need to extend an existing class to create a 
data loader. A ll we need is an __iter__ method to retrieve batches of data of an __len__ method to the number of 
batches"""
class DeviceDataLoader(): 
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device 

    def __iter__(self): 
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device) 

    def __len__(self): 
        """Number of batches"""
        return len(self.dl)
    
"""The yeild keyword in python is used to create a generator function that can be used within a for loop, as 
illustrated below"""
def some_numbers(): 
    yield 10
    yield 20 
    yield 30  

for value in some_numbers(): 
    #print(value)
    pass


"""We can now wrap our data loaders using DeviceDataLoader""" 
train_dl = DeviceDataLoader(train_dl, device) 
val_dl = DeviceDataLoader(val_dl, device) 

"""Tensors moved to the GPU have a device property which includes the word cuda. Let's verify this by 
looking at a batch of val_dl"""
for xb, yb in val_dl: 
    #print(f"xb device: {xb.device}")
    #print(f"yb: {yb}")
    break 

"""Training the Model
We'll define two functions: fir and evaluate to train the model using gradient descent and evaluate 
its performace on the validation set."""
def evaluate(model, val_dl): 
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_dl] 
    return model.validation_epoch_end(outputs) 

def fit(epochs, lr, model: nn.Module, train_dl, val_dl, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = [] 
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs): 
        "Training phase" 
        for batch in train_dl: 
            loss = model.training_step(batch) 
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 
        
        # Validation phase 
        result = evaluate(model, val_dl) 
        model.epoch_end(epoch, result) 
        history.append(result)
    return history

"""Before we train the model, we need to ensure that the data and the model's parameters (weights and biases) 
are on the same device (CPU or GPU). We can reuse the to_device function to move the model's parameters to the right 
device"""

# Model (on GPU) 
model = MnistModel(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
to_device(model, device) 

"""Let's see how the model performs on the validation set with the initial set of weights"""
history = [evaluate(model, val_dl)] 
print(history)

"""Let's train the model for five epochs and look at the result"""
history = fit(7, 0.5, model, train_dl, val_dl) 

"""Let's plot the losses and accuracies to study how the model improves"""
losses = [x["val_loss"] for x in history]
accs = [x["val_acc"] for x in history] 

plt.figure(figsize=(10, 10))
plt.subplot(1,2, 1) 
plt.plot(losses, "-x")
plt.xlabel("epoch")
plt.ylabel("loss") 
plt.title("Loss vs No. of epochs")

plt.subplot(1, 2, 2) 
plt.plot(accs, "-x") 
plt.xlabel("epoch")
plt.ylabel("accuracy") 
plt.title("Accuracy vs. No. of epochs") 
#plt.show()

"""Our current model outperforms the logistic regression model (which could only achieve around 90%) by 
a considerable margin. To improve further, we need to make the model more powerful 
by increasing the hidden layers's size or adding more hidden layers with 
activations. """

"""Testing individual images 
While we have been tracking the overall accuracy of a model for far, it's also
a good idea to look at the model's results on some sample images. Let's test out our 
model with some images from the pre-defined test dataset of 10_000 images.""" 
def predict_image(img, model): 
    xb = to_device(img.unsqueeze(0), device) 
    yb = model(xb) 
    preds = torch.argmax(yb, dim=1)  
    return preds[0].item() 

"""Let's try it out with a few images""" 
img, label = test_dataset[0] 
plt.figure() 
plt.imshow(img[0], cmap="gray")
plt.title(f"Label: {label} | Predicted: {predict_image(img, model)}") 

img, label = test_dataset[1839] 
plt.figure() 
plt.imshow(img[0], cmap="gray")
plt.title(f"Label: {label} | Predicted: {predict_image(img, model)}")


img, label = test_dataset[193] 
plt.figure() 
plt.imshow(img[0], cmap="gray")
plt.title(f"Label: {label} | Predicted: {predict_image(img, model)}")
plt.show()

