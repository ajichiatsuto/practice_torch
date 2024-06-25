import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = AlexNet(10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    num_epochs = 10
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in num_epochs:
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            train_loss += loss.item()
            train_acc += (output.max(1)[1]==labels).sum().item()

            loss.backward()
            optimizer.step()
        avg_train_loss = train_loss/len(train_loader)
        avg_train_acc = train_acc/len(train_loader)

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                ouput = model(images)
                loss = criterion(output, labels)

                val_loss += loss.item()
                val_acc += (ouput.max(1)[1]==labels).sum().item()
        avg_val_loss = val_loss/len(test_loader)
        avg_val_acc = val_acc/len(test_loader)

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'.format(epoch+1, num_epochs, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
    
    plt.figure()
    plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
    plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='-', label='train_acc')
    plt.legend()
    plt.grid()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss in train and val")
    plt.show()

    plt.figure()
    plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
    plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='-', label='val_acc')
    plt.legend()
    plt.grid()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("accuracy in train and val")
    plt.show()
