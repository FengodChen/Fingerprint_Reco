import argparse
import netMethod
import imgOperator

imageReader = imgOperator.ImageReader('./img')
(trainImg_numpy, trainLabel_numpy) = imageReader.getSet(4, False)

trainImg_tensor = netMethod.numpy2tensor(trainImg_numpy)
trainLabel_tensor = netMethod.numpy2tensor(trainLabel_numpy)
trainData_tensor = (trainImg_tensor, trainLabel_tensor)
#trainer = netMethod.Trainer('./save.pth', 50)
trainer = netMethod.Trainer('./save.pth', 50, './save.pth')
trainer.train(trainData_tensor)