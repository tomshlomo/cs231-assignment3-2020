import torch
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from cs231n.net_visualization_pytorch import preprocess, deprocess, rescale, blur_image

# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False

from cs231n.data_utils import load_imagenet_val

X, y, class_names = load_imagenet_val(num=5)

from cs231n.net_visualization_pytorch import make_fooling_image

idx = 0
target_y = 6

X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
X_fooling = make_fooling_image(X_tensor[idx:idx + 1], target_y, model)

scores = model(X_fooling)
print(torch.softmax(scores, dim=1)[0, target_y])
if not target_y == scores.data.max(1)[1][0].item():
    print('The model is not fooled!')
