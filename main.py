import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import imgOperator

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

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x.float())))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 92 * 92)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, trainNum, data):
        (inData, outData) = data
        for i in range(trainNum):
            self.optimizer.zero_grad()
            output = self(inData)
            print(output.shape)
            print(outData.shape)
            l = self.loss(output, outData)
            l.backward()
            self.optimizer.step()

            if (i % 100 == 0):
                print("Epoch: {}, loss = {}".format(i, l.item()))

if __name__ == "__main__":
    net = Net()

    imageReader = imgOperator.ImageReader('./img')
    (trainImg_numpy, trainLabel_numpy) = imageReader.getSet(4, False)
    #trainImg_tensor = torch.tensor(trainImg_numpy)
    #trainLabel_tensor = torch.tensor(trainLabel_numpy)
    trainImg_tensor = torch.from_numpy(trainImg_numpy)
    trainLabel_tensor = torch.from_numpy(trainLabel_numpy)
    trainData_tensor = (trainImg_tensor, trainLabel_tensor)

    net.train(1000, trainData_tensor)