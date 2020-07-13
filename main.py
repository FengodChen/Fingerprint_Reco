import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import imgOperator

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1))
        x = self.pool2(F.relu(self.conv2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, trainNum, data):
        (inData, outData) = data
        for i in range(trainNum):
            self.optimizer.zero_grad()
            output = self(inData)
            l = self.loss(output, outData)
            l.backward()
            self.optimizer.step()

            if (i % 100 == 0):
                print("Epoch: {}, loss = {}".format(i, l.item()))

if __name__ == "__main__":
    net = Net()
    imageReader = imgOperator.ImageReader('./img')
    trainData = imageReader.getSet(4, False)
    net.train(1000, trainData)