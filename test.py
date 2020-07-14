import argparse
import netMethod
import imgOperator

imageReader = imgOperator.ImageReader('./img')
(testImg_numpy, testLabel_numpy) = imageReader.getSet(1, True)

testImg_tensor = netMethod.numpy2tensor(testImg_numpy)
testLabel_tensor = netMethod.numpy2tensor(testLabel_numpy)
testData_tensor = (testImg_tensor, testLabel_tensor)
tester = netMethod.Tester('./save.pth')
tester.test(testData_tensor)