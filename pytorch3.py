import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True,
                            transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                            transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #initialize our neural net
    def __init__(self):
        super().__init__()
        # fc is fully connected (our nodes)
        # Linear takes input and output layers
        # 784 inputs from flatten image and goes into 64 neuron nodes
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        #last one requires the output nodes (0-9) for our classifier
        self.fc4 = nn.Linear(64, 10)

    #how we pass data to the neural net
    def forward(self, x):
        #relu is a rectified Linear Unit which is our sigmoid to determind if
        #the neuron should be fired up (Activated) This is our Activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        #dim is our distribution we want to solve to one
        return F.log_softmax(x, dim=1)

net = Net()
print(net)

X = torch.rand((28*28))
X = X.view(1,28*28)

output = net(X)
print(output)


import torch.optim as optim
#optimizing the rate of how our model learns
optimizer = optim.Adam(net.parameters())
