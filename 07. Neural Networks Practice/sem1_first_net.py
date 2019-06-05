import warnings

warnings.filterwarnings("ignore")

import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

transform = transforms.Compose(

    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

)

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                           shuffle=True, num_workers=0, pin_memory=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
validation_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=0, pin_memory=False)
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
BATCH_IMAGE_COUNT = 10000
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
N_CLASSES = len(CLASSES)
PLOT = False


class Net(torch.nn.Module):
    def __init__(self, n_hidden_nodes, n_hidden_layers):
        super(Net, self).__init__()
        # Определяем слои
        self.fc1 = torch.nn.Linear(IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS,
                                   n_hidden_nodes)
        #         self.n_hidden_nodes=n_hidden_nodes
        #         self.n_hidden_layers=n_hidden_layers

        self.out = torch.nn.Linear(n_hidden_nodes, N_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS)
        # определяем функцию активации
        sigmoid = torch.nn.Sigmoid()
        x = sigmoid(self.fc1(x))
        return torch.nn.functional.log_softmax(self.out(x))


def train(epoch, model, train_loader, optimizer, log_interval=100, cuda=None):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        accuracy = 100. * correct / len(train_loader.dataset)

        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data, accuracy))


def validate(loss_vector, accuracy_vector, model, validation_loader, cuda=None):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += torch.nn.functional.nll_loss(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


hidden_nodes = 10
layers = 1
cuda = None
model = Net(hidden_nodes, layers)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_vector = []
acc_vector = []
for epoch in range(1, EPOCHS + 1):
    train(epoch, model, train_loader, optimizer, cuda=cuda)
    validate(loss_vector, acc_vector, model, validation_loader, cuda=cuda)
    if epoch == 40:
        torch.save(model.state_dict(), 'Second_Net.pt')
        break
