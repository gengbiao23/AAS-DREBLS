
import numpy as np
import scipy.io as scio
from BroadLearningSystem import BLS


dataFile = './mnist.mat'
data = scio.loadmat(dataFile)
traindata = np.double(data['train_x']/255)
trainlabel = np.double(data['train_y'])
testdata = np.double(data['test_x']/255)
testlabel = np.double(data['test_y'])

# Demo4BLS.py
traindata = traindata[:1000]  # 先用1000样本测试
trainlabel = trainlabel[:1000]

N1 = 10  #  # of nodes belong to each window
N2 = 10  #  # of windows -------Feature mapping layer
N3 = 500 #  # of enhancement nodes -----Enhance layer
L = 5    #  # of incremental steps 
M1 = 50  #  # of adding enhance nodes
s = 0.8  #  shrink coefficient
C = 2**-30 # Regularization coefficient

print('-------------------DREBLS---------------------------')
BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)












