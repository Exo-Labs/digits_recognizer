import torch
import torch.nn as nn
import torch.optim as optim #optimization functions
import torch.nn.functional as F #activation functions
from torch.utils.data import DataLoader #data load and preparation
import torchvision.datasets as datasets #ready datasets
import torchvision.transforms as transforms #modification of dataset

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 500
num_classes = 10
lr = 0.001
batch_size = 64
iterations = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# transforms.ToTensor()

training_set = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
training_loader = DataLoader(dataset = training_set, batch_size = batch_size, shuffle=True)
test_set = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes, hidden_size=hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for i in range(iterations):
    for batch_idx, (data, targets) in enumerate(training_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)
        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader, mode):
    if loader.dataset.train:
        print('Training data')
    else:
        print('Test data')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += (predictions.size(0))

    accur = float(num_correct/num_samples * 100)

    print('Acuracy rate: ',  accur)

check_accuracy(training_loader, model)
check_accuracy(test_loader, model)

torch.save(model.state_dict, 'mnist_model.pth')