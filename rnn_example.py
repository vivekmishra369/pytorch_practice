import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

# make network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # same convolution
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) 
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # N*time_seq*time_features
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #Forward prop
        out,_ = self.rnn(x,h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

# model = NN(784,10)
# x = torch.randn(64,784)
# print(model(x))

# model = CNN()
# x = torch.randn(64,1,28,28)
# print(model(x).shape)
# exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
input_size = 28 
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001 
batch_size = 64 
num_epochs = 1

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=1)
test_dataset = datasets.MNIST(root='dataset/', train=0, transform=transforms.ToTensor(), download=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=1)
 
# init network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# laoss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# train
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        # print(batch_idx, data.shape, targets.shape)
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        # get it to correct shape
        # data = data.reshape(data.shape[0],-1)
        scores = model(data)
        loss = criterion(scores,targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

# check performance
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on trianing data")
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # print("scores.shape", scores.shape)
            _, predictions = scores.max(1)
            num_correct += (predictions ==y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)