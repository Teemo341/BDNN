import torch
import torch.nn as nn
import torch.nn.init as init
__all__ = ['Rank1BNN']

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

class Rank1BNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, rank=1):
        super(Rank1BNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        
        # Rank-1 factorized weights
        self.u = nn.Parameter(torch.Tensor(out_channels, rank))
        self.v = nn.Parameter(torch.Tensor(rank, in_channels, kernel_size, kernel_size))
        
        # Bias term
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.u)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = torch.einsum('oi,icjk->ocjk', self.u, self.v)
        return torch.nn.functional.conv2d(x, weight, self.bias, stride=1, padding=1)

class Rank1ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, rank=1):
        super(Rank1ResBlock, self).__init__()
        self.conv1 = Rank1BNNLayer(in_channels, out_channels, stride=stride, rank=rank)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Rank1BNNLayer(out_channels, out_channels, rank=rank)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Rank1BNN(nn.Module):
    def __init__(self, rank=1):
        super(Rank1BNN, self).__init__()
        self.downsampling_layers = nn.Sequential(
            Rank1BNNLayer(3, 64, 3, 1, rank=rank),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Rank1BNNLayer(64, 128, 4, 2, 1, rank=rank),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Rank1BNNLayer(128, 256, 4, 2, 1, rank = rank),
        )
        self.feature_layers = [Rank1ResBlock(256, 256, rank=rank) for _ in range(6)]
        self.fc_layers = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 200))

        self.model = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)

    def forward(self, x):
        out = self.model(x)
        return out

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Hyperparameters
num_classes = 10
rank = 1
batch_size = 64
learning_rate = 0.01
epochs = 10

# Data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, optimizer and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Rank1ResNet18(Rank1ResBlock, [2, 2, 2, 2], num_classes, rank).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training and testing the model
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test(model, device, test_loader, criterion)


def test():
    model = Resnet()
    return model  
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = test()
    num_params = count_parameters(model)
    print(num_params)