import os
import numpy as np
import cv2 as cv

class ImageReader():
    def __init__(self, path:str):
        self.path = path
        self.tree = {}
        for dirName in os.listdir(path):
            self.tree[dirName] = os.listdir("{}/{}".format(path, dirName))

        self.classesNum = len(self.tree)
    
    def makeOneHot(self, label:int):
        onehot = np.zeros(self.classesNum, dtype=np.float)
        onehot[label] = 1
        return onehot

    def getSet(self, num:int, invert:bool):
        '''
        if (invert==false) {
            get array[:num]
        } else {
            get array[-num:]
        }

        return (dataset, label)
        '''
        imgList = []
        labelList = []
        for dirName in self.tree:
            label = int(dirName[3:])
            onehotLabel = self.makeOneHot(label)
            
            imgTree = None
            if (invert):
                imgTree = self.tree[dirName][-num:]
            else:
                imgTree = self.tree[dirName][:num]
            
            for imgName in imgTree:
                imgPath = os.path.join(self.path, dirName, imgName)
                img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
                #img = cv.imread(imgPath)
                imgList.append(np.array([img/255.], dtype=np.float))
                #img.resize(3, 200, 200)
                #imgList.append(img)
                labelList.append(onehotLabel)
        
        imgList = np.array(imgList, dtype=np.double)
        labelList = np.array(labelList, dtype=np.double)
        return (imgList, labelList)

if __name__ == '__main__':
    imageReader = ImageReader('./img')
    (trainImg_numpy, trainLabel_numpy) = imageReader.getSet(4, False)
    (testImg_numpy, testLabel_numpy) = imageReader.getSet(1, True)
    print(trainLabel_numpy)
    for i in testLabel_numpy:
        print(i)