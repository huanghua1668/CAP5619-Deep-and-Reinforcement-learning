'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import tensorflow as tf
#from tensorflow import keras
import  keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1./(1.+np.exp(-x))

def prob1():
    z=np.arange(-10, 10, 0.01)
    df_reLU=np.zeros_like(z)
    df_reLU[z>=0.]=1.

    df_sigmoid=np.zeros_like(z)
    df_sigmoid=sigmoid(z)*(1.-sigmoid(z))
    
    df_linear=np.ones(z.shape[0])*0.05
    df_linear[(abs(z)<=1.)]=1.

    df_swish=np.zeros_like(z)
    df_swish=sigmoid(5.*z)*(1.+5*z*(1.-sigmoid(5*z)))

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(z, df_reLU, label='reLU')
    plt.plot(z, df_sigmoid, label='sigmoid')
    plt.plot(z, df_linear, label='piecewise linear')
    plt.plot(z, df_swish, label='swish')
    plt.xlabel('z')
    plt.ylabel('derivative')
    plt.legend()
    plt.xlim([-10, 10])
    plt.ylim([-0.5, 1.5])
    plt.show()

def g(x):
    return 1.+np.sin(1.5*np.pi*x)

def max_deviation(y_true, y_pred):
    return K.max(np.abs(y_true-y_pred))

def prob3():
    epochs = 2000
    sz=100
    x_train=np.zeros(sz)
    x_test=np.zeros(sz)
    m=sz
    
    np.random.seed(13751)
    for i in range(sz):
        x_train[i]=np.random.uniform(-2., 2.)
    y_train=g(x_train)
    
    x_train=x_train.reshape(m,1)
    
    x_train= x_train.astype('float32')
    print(x_train.shape[0], 'train samples')
    
    model = Sequential()
    n=32
    model.add(Dense(n, activation='relu', input_shape=(1,)))
    #hh: here use_bias=False, so no bias
    #model.add(Dropout(0.2))
    #hh: Dropout consists in randomly setting a fraction rate of input
    #    units to 0 at each update during training time, which helps prevent
    #    overfitting.
    model.add(Dense(n, activation='relu'))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(n, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.summary()
    
    model.compile(loss='mse',
                  optimizer=RMSprop(),
                  metrics=['mae', max_deviation])
    
    batch_size = 16
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    
    x_train=np.sort(x_train[:,0])
    #print('x_train', x_train)
    y_train=g(x_train)
    preds=model.predict(x_train)
    plt.plot(x_train, y_train, label='True')
    plt.plot(x_train, preds, 'o', label='prediction')
    plt.ylabel('y')
    plt.legend()
    plt.xlabel('x')
    print('max deviation:', np.max(np.absolute(y_train-preds[:,0])))
    plt.show()

def prob4():
    loaded=np.loadtxt('hw2_softmax_weights.m')
    bias=loaded[0]
    weight=loaded[1:]
    loaded=np.genfromtxt('hw2_softmax_sample.txt', delimiter=',')
    x=loaded
    print('bias', bias.shape)
    print('weight', weight.shape)
    print('sample x', x.shape)
    print('x', x)
    
    z=weight.T.dot(x)+bias
    z0=np.copy(z)
    print('z', z.shape)
    print('z', z)
    expZ=np.exp(z)
    softmax=expZ/np.sum(expZ)
    print('softmax z', softmax)
    index=np.argsort(softmax)
    pred=index[-1]
    print('class: ', pred)
    
    p=np.zeros(20)
    p[0]=1.
    q=np.copy(softmax)
    dw=np.outer(x, q-p)
    dBias=np.copy(q-p)
    print('dw', dw.shape)
    print('q-p', q-p)
    #for i in range(dw.shape[0]):
    #    print(dw[i])
    print('positive dw, and decreased weights will be', np.sum(dw>=0.01))
    print('negative dw, and increased weights will be', np.sum(dw<=-0.01))
    print('small dw(abs<0.01), and no change weights will be', np.sum(np.absolute(dw)<0.01))
    print('positive dBias, and decreased bias will be', np.sum(dBias>=0.01))
    print('negative dBias, and increased bias will be', np.sum(dBias<=-0.01))
    print('small dBias(abs<0.01), and no change bias will be', np.sum(np.absolute(dBias)<0.01))
    
    temp=weight.T
    w0=temp[1]
    dx=w0/np.linalg.norm(w0)
    c=[0., 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    for i in range(temp.shape[0]):
        l=np.linalg.norm(temp[i])
        print(l)
    
    for i in range(len(c)):
        newX=x+c[i]*dx
        z=weight.T.dot(newX)+bias-z0
        print('diff z for ', c[i], z[1])
        pred=np.argsort(z)
        #print('class', pred)
        print('z', z)

#prob1() 
#prob3() 
prob4() 
