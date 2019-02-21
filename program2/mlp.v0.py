'''Trains a simple deep NN on the sin function.
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

H=2 # number of hidden layer neurons
model = {}
#model['W1'] = np.random.randn(H, 2)/ np.sqrt(H)# "Xavier" initialization
#model['W2'] = np.random.randn(1, H+1) / np.sqrt(H+1)#hhuang:shape (H, )
model['W1'] = np.array([[1., 2],[3, 4]])
model['W2'] = np.array([[5., 6, 7]])

def g(x):
    return 1.+np.sin(1.5*np.pi*x)

def forward(X):
    print('forward')
    h=np.dot(X, model['W1'].T)
    print('x', X)
    print('w1', model['W1'])
    print('h', h)
    h[h<0]=0
    print('reLU, h', h)
    h=np.insert(h, 0, 1., axis=1)# add a bias unit
    print('add bias, h', h)
    print('w2', model['W2'])
    val=np.dot(h, model['W2'].T)
    print('val', val)
    return val, h

def backward(X, h, err):
    print('backward')
    print('X', X)
    print('h', h)
    print('err', err)
    m=h.shape[0]
    dW2=-err.T .dot(h)/m
    print('dW2', dW2)
    dh=-np.outer(err, model['W2'])
    print('dh', dh)
    dh[h<=0]=0.
    print('dh after reLU', dh)
    dh=dh[:,1:]
    print('dh eliminate bias', dh)
    dW1=dh.T .dot(X)/m
    print('dW1', dW1)
    return dW1, dW2

epochs = 2

sz=2
x_train=np.zeros(sz)
x_test=np.zeros(sz)
m=sz
for i in range(sz):
    x_train[i]=np.random.uniform(-2., 2.)
    x_test[i]=np.random.uniform(-2., 2.)
x_train=np.array([1./9, 1./3])
y_train=g(x_train)
y_test=g(x_test)

x_train1=x_train.reshape(m,1)
x_test1=x_test.reshape(m,1)
X=np.insert(x_train1, 0, 1., axis=1)
XTest=np.insert(x_test1, 0, 1., axis=1)

print('x train', x_train)
print('y train', y_train)
alpha=0.001
mse=np.zeros(epochs)
mseTest=np.zeros(epochs)
for i in range(epochs):
    val, h=forward(X)
    err=y_train.reshape(sz, 1)-val
    valTest, h_=forward(XTest)
    errTest=y_test-valTest
    dW1, dW2=backward(X, h, err)
    model['W1']-=alpha*dW1
    model['W2']-=alpha*dW2
    mse[i]=np.linalg.norm(err)/np.sqrt(m)
    print(i, 'max difference: ', np.max(np.absolute(err)))
    mseTest[i]=np.linalg.norm(errTest)/np.sqrt(m)
    print(i, 'err, test err: ', mse[i], mseTest[i])

    if(i==epochs-1):
        figureIndex=0
        plt.figure(figureIndex)
        x=np.sort(x_train)
        print('x', x)
        index=np.argsort(x_train)
        y=np.zeros(m)
        yTrue=np.zeros(m)
        for j in range(sz):
            y[j]=val[index[j]]
            yTrue[j]=y_train[index[j]]
        plt.plot(x, y, 'c', label='predicted')
        print('y:', y)
        plt.plot(x, yTrue, label='true')
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.ylim([0, 0.25])
        plt.legend()

figureIndex=1
plt.figure(figureIndex)
figureIndex+=1
plt.plot(np.arange(epochs), mse, label='train')
plt.plot(np.arange(epochs), mseTest, label='test')
plt.xlabel('Epochs')
plt.ylabel('MSE')
#plt.ylim([0, 0.25])
plt.legend()
plt.show()
