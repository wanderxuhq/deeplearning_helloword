import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

random_seed = 1
torch.manual_seed(random_seed)

batch_size_train = 64
batch_size_test = 1000

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST(
    './data/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size_train, shuffle=True)

test_data = torchvision.datasets.MNIST(
    './data/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size_test, shuffle=False)

print(train_loader)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x= x.view(-1, 784)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x

model = Net()

learning_rate = 0.01
momentum = 0.2
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)

lost_fucnction = nn.CrossEntropyLoss()

if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))

def train():
    for index, data in enumerate(train_loader):
        input, target = data
        optimizer.zero_grad()
        y_predictor = model(input)
        loss = lost_fucnction(y_predictor, target)
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print(loss.item())
            torch.save(model.state_dict(), './model/model.pkl')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')

def test():
    total = 0
    correct = 0
    with torch.no_grad():
        for index, data in enumerate(test_loader):
            input, target = data
            output = model(input)
            _, index = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (index == target).sum().item()

    print(correct / total)

if __name__ == '__main__':
    for i in range(5):
        train()
        test()