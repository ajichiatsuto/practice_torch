import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(in_features=10 ,out_features=10, bias=False)
    
    def forward(self, x):
        return self.lin1(x)

def main(opt_name):
    loss_list = []

    x = torch.randn(1,10)
    w = torch.randn(1,1)
    y = torch.mul(x,w) + 2

    model = Model()
    criterion = nn.MSELoss()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif opt_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    elif opt_name == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)
    elif opt_name == "momentum_sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(20):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data.item())
    
    return loss_list

losss_dict = {}
losss_dict["sgd"] = main("sgd")
losss_dict["adam"] = main("adam")
losss_dict["adagrad"] = main("adagrad")
losss_dict["rmsprop"] = main("rmsprop")
losss_dict["adadelta"] = main("adadelta")
losss_dict["momentum_sgd"] = main("momentum_sgd")

plt.figure()
for key in losss_dict.keys():
    plt.plot(losss_dict[key], label=key)

plt.legend()
plt.grid()
plt.show()