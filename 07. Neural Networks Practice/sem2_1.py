import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch

import torchvision
from torchvision import transforms
from PIL import Image
from IPython import display
from sklearn.metrics import accuracy_score
from tqdm import tqdm


classes = os.listdir("./notMNIST_train")
n_classes = len(classes)
classes = sorted(classes)
IMG_H = 28
IMG_W = 28

batch_size = 64

train_data = torchvision.datasets.ImageFolder('./notMNIST_train/', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])
    )
train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = torchvision.datasets.ImageFolder('./notMNIST_val/', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])
    )
val_loader = data_utils.DataLoader(val_data, batch_size=1, shuffle=False)

hid_size = 100
out_size = n_classes


class TwoLayerNet(nn.Module):
    def __init__(self, h, w, hid_size, out_size):
        super(TwoLayerNet, self).__init__()
        # объявляем слои для нашей сети
        self.linear1 = nn.Linear(h * w, hid_size)
        self.linear2 = nn.Linear(hid_size, out_size)

    def forward(self, x):
        x = self.linear1(x)
        # функция активации
        x = F.relu(x)
        x = self.linear2(x)
        return x



two_layer_net = TwoLayerNet(h=IMG_H, w=IMG_W, hid_size=hid_size, out_size=out_size)
loss_fn = torch.nn.modules.loss.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.SGD(two_layer_net.parameters(), lr=learning_rate)
def train(train_loader, val_loader, num_epochs=100):
    train_losses = []
    val_losses = []

    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        display.clear_output(wait=True)

        # 1. forward
        two_layer_net.train(True)
        loss_batch = []
        acc_batch = []
        for x_train, y_train in train_loader:
            x_train = Variable(x_train.view(x_train.shape[0], -1))
            # 1.1 получаем предсказания сети
            y_pred = two_layer_net(x_train)
            # 1.2 вычисляем accuracy на данном батче по предсказаниями и правильным ответам
            acc_batch.append(accuracy_score(np.argmax(y_pred.data.cpu().numpy(), axis=1), y_train.numpy()))
            # 1.3 вычисляем loss (кросс-эетропию)
            loss_train = loss_fn(y_pred, Variable(y_train))
            # 1.4 backward
            loss_train.backward()
            # 1.5 обновляем
            optimizer.step()
            # 1.6 зануляем
            optimizer.zero_grad()
            # 1.7 запоминаем
            loss_batch.append(loss_train.data.cpu().numpy())

        train_losses.append(np.mean(loss_batch))
        train_acc.append(np.mean(acc_batch))

        # 2. val
        two_layer_net.train(False)
        loss_batch = []
        acc_batch = []
        for i, (x_val, y_val) in enumerate(val_loader):
            x_val = Variable(x_val.view(x_val.shape[0], -1))
            # 2.1 получаем предсказания сети
            y_pred = two_layer_net(x_val)
            acc_batch.append(accuracy_score(np.argmax(y_pred.data.cpu().numpy(), axis=1), y_val.numpy()))
            # 2.3 вычисляем loss (кросс-эетропию)
            loss_val = loss_fn(y_pred, Variable(y_val))
            # 2.4 запоминаем
            loss_batch.append(loss_val.data.cpu().numpy())

        val_losses.append(np.mean(loss_batch))
        val_acc.append(np.mean(acc_batch))

        # 3. будем сохранять модель в файл, если после этой эпохи loss упал
        torch.save(two_layer_net.state_dict(), 'Seminar_2_1.pt')

        # 4. show plot
        _, axes = plt.subplots(1, 2, figsize=(16, 6))

        plt.title("losses")
        axes[0].plot(train_losses, label="train loss")
        axes[0].plot(val_losses, label="val loss")
        plt.legend()

        plt.title("accuracies")
        axes[1].plot(train_acc, label="train accuracy")
        axes[1].plot(val_acc, label="val accuracy")
        plt.legend()

        plt.show()

        print("Final loss: ", val_losses[-1])
        print("Final accuracy: ", val_acc[-1])

train(train_loader, val_loader, num_epochs=20)