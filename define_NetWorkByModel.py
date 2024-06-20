import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    # __init__でレイヤーを定義
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 64, 5)
    
    # forwardで順伝播の計算を定義
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

model = Model()

