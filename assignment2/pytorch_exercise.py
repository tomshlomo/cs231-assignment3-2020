import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from ResNet import ResNet, ResNetBasicBlock


NUM_TRAIN = 49000
NUM_VAL = 1000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=False,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=False,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))

# cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
#                             transform=transform)
# loader_test = DataLoader(cifar10_test, batch_size=64)

# You have an option to **use GPU by setting the flag to True below**. It is not necessary to use GPU for this
# assignment. Note that if your computer does not have CUDA enabled, `torch.cuda.is_available()` will return False
# and this notebook will fallback to CPU mode.

# The global variables `dtype` and `device` will control the data types throughout this assignment.

USE_GPU = False

dtype = torch.float32  # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 10

print('using device:', device)

num_classes = 100
in_channel = 3

writer = SummaryWriter('runs/CIFAR_experiment_1')

# %% helper functions
def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                print('Validation accuracy:')
                check_accuracy_part34(loader_val, model)
                # print('Training accuracy:')
                # check_accuracy_part34(loader_train, model)
                print()


def check_accuracy_part34(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


# %% Toy model with 2 convolution layers
# channel_1 = 32
# channel_2 = 16
# # learning_rate = 1e-2
# conv1 = nn.Conv2d(in_channel, channel_1, (5, 5), padding=2)
# conv2 = nn.Conv2d(channel_1, channel_2, (3, 3), padding=1)
# fc = nn.Linear(32 * 32 * channel_2, num_classes)
# nn.init.kaiming_normal_(conv1.weight)
# nn.init.kaiming_normal_(conv2.weight)
# nn.init.kaiming_normal_(fc.weight)
# model = nn.Sequential(
#     conv1,
#     nn.ReLU(),
#     conv2,
#     nn.ReLU(),
#     Flatten(),
#     fc)

# optimizer = optim.SGD(model.parameters(), lr=learning_rate,
#                       momentum=0.9, nesterov=True)

model = ResNet(in_channel, num_classes, block=ResNetBasicBlock, depths=[2, 2], block_sizes=[8, 16])
dataiter = iter(loader_train)
images, labels = dataiter.next()
# writer.add_graph(model, images)
learning_rate = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_part34(model, optimizer, epochs=1000)
