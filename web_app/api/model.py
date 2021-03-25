import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as transforms 
from PIL import Image, ImageOps

# load model

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
        x = self.l1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.l2(x)
        return x

num_classes = 10
in_channels = 1

model = CNN(in_channels=in_channels, num_classes=num_classes)

PATH = "CNN_model.pth"
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28,28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))
                                    ])

    image = Image.open(image_bytes)
    image = image.resize((28, 28))
    image = ImageOps.invert(image.convert('RGB'))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    # For NN
    # images = image_tensor.reshape(-1, 28*28)
    
    # For CNN
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
