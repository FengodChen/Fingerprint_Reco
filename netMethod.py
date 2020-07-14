import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)     # in: 200*200*3     out: 196*196*6
        self.pool1 = nn.MaxPool2d(2, 2)     # in: 196*196*6     out: 98*98*6
        self.conv2 = nn.Conv2d(6, 16, 5)    # in: 98*98*6       out: 94*94*16
        self.pool2 = nn.MaxPool2d(3, 1)     # in: 94*94*16      out: 92*92*16
        self.fc1 = nn.Linear(92*92*16, 120) # in: 92*92*16      out: 120
        self.fc2 = nn.Linear(120, 84)       # in: 120           out: 84
        self.fc3 = nn.Linear(84, 50)        # in: 84            out: 50

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x.float())))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 92 * 92)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.normalize(self.fc3(x))
        return x

class Trainer:
    def __init__(self, save_file:str, save_epochs:int, load_file:str=None):
        self.save_file = save_file
        self.save_epochs = save_epochs
        self.net = Net()

        if (not load_file==None):
            self.net.load_state_dict(torch.load(load_file))
            self.net.eval()

    def train(self, data):
        (inData, outData) = data
        i = 1

        while (True):
            self.net.optimizer.zero_grad()
            output = self.net(inData)
            l = self.net.loss(output.double(), outData.double())
            l.backward()
            self.net.optimizer.step()

            print("Epoch: {}, loss = {}".format(i, l.item()))
            
            if (i % self.save_epochs == 0):
                torch.save(self.net.state_dict(), self.save_file)
            i += 1

class Tester:
    def __init__(self, load_file:str):
        self.net = Net()
        self.net.load_state_dict(torch.load(load_file))
        self.net.eval()

    def test(self, data):
        (inData, outData) = data
        output = self.net(inData)

        trueArray = []
        predictArray = []

        length = len(outData)
        for ptr in range(length):
            trueArray.append(int(torch.argmax(outData[ptr])))
            predictArray.append(int(torch.argmax(output[ptr])))
            print(output[ptr])

        print(trueArray)
        print(predictArray)
        
        trueArray = np.array(trueArray, dtype=np.uint32)
        predictArray = np.array(predictArray, dtype=np.uint32)

def numpy2tensor(np_array):
    return torch.from_numpy(np_array)
