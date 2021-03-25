import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #optimization functions
from torch.utils.data import DataLoader #data load and preparation
import torchvision.datasets as datasets #ready datasets
import torchvision.transforms as transforms #modification of dataset

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.l1 = nn.Linear(32 * 3 * 3, 500)
        self.l2 = nn.Linear(500, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = self.dropout2(x)
        x = self.l2(x)
        return x

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

training_set = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
training_loader = DataLoader(dataset = training_set, batch_size = batch_size, shuffle=True)
test_set = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle=True)

in_channels = 1
num_classes = 10
lr = 0.001
iterations = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for i in range(iterations):
    for batch_idx, (data, targets) in enumerate(training_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        scores = model(data)
        loss = criterion(scores, targets)
        loss.backward()
        
        optimizer.zero_grad()
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
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += (predictions.size(0))

    accur = float(num_correct/num_samples * 100)

    print('Acuracy rate: ',  accur)

check_accuracy(training_loader, model)
check_accuracy(test_loader, model)

torch.save(model.state_dict(), 'CNN_model.pth')