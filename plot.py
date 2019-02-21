#######################################################################
# Copyright (C)                                                       #
# 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

figureIndex=0
def myPlot(fileName):
    results=np.loadtxt(fileName)
    epos=results[:,0]
    epos=epos.astype('int')
    lossTrain=results[:,1]
    accuracyTrain=results[:,2]
    lossTest=results[:,3]
    accuracyTest=results[:,4]
    global figureIndex
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(epos, lossTrain, label='train')
    plt.plot(epos, lossTest, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.ylim([0, 0.25])
    plt.legend()

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(epos, accuracyTrain, label='train')
    plt.plot(epos, accuracyTest, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.ylim([0, 1.])
    plt.legend()


if __name__ == '__main__':
    myPlot('data1.dat')
    myPlot('data2.dat')
    plt.show()




