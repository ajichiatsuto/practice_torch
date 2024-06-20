import torch.nn as nn
from collections import OrderedDict

model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)

model2 = nn.Sequential()
model2.add_module('conv1', nn.Conv2d(1, 20, 5))
model2.add_module('relu1', nn.ReLU())
model2.add_module('conv2', nn.Conv2d(20, 64, 5))
model2.add_module('relu2', nn.ReLU())

model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 20, 5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20, 64, 5)),
    ('relu2', nn.ReLU())
]))