from __future__ import print_function
import tensorflow as tf
#from tensorflow import keras
import  keras
#from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, Average
from keras.layers import Activation, MaxPooling2D
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras import backend as K

def make_mosaic(imgs, nrows, ncols, layer, border=1):
    # in reference to
    # https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    """ Given a set of images with all the same shape, makes a
        mosaic with nrows and ncols"""
    if layer==0: # images 2*4
        nimgs = imgs.shape[-1]
        imshape = imgs.shape[:-1]
        print('imgs shape', imgs.shape)
        print('imshape', imshape)
            
        mosaic = np.ma.masked_all((nrows * imshape[0]+ (nrows - 1) *border,
                                   ncols * imshape[1]+ (ncols- 1)*border),
                                   dtype=np.float32)
        paddedh = imshape[0] + border
        paddedw = imshape[1] +border
        for i in range(nimgs):
            row =int(np.floor(i/ncols))
            col=i%ncols
        
            mosaic[row*paddedh:row*paddedh+imshape[0],
                    col*paddedw:col*paddedw+imshape[1]]=imgs[:,:,i]
    else:
        nimgs = imgs.shape[-1]*imgs.shape[-2]
        imshape = imgs.shape[:-2]
        print('imgs shape', imgs.shape)
        print('imshape', imshape)
            
        mosaic = np.ma.masked_all((nrows * imshape[0]+ (nrows - 1) *border,
                                   ncols * imshape[1]+ (ncols- 1)*border),
                                   dtype=np.float32)
        paddedh = imshape[0] + border
        paddedw = imshape[1] +border
        for i in range(nimgs):
            row =int(np.floor(i/ncols))
            col=i%ncols
        
            mosaic[row*paddedh:row*paddedh+imshape[0],
                    col*paddedw:col*paddedw+imshape[1]]=imgs[:,:,row,
                            col]
    return mosaic

def cnn():
    batch_size = 128
    #batch_size = 4096
    num_classes = 10
    epochs = 5
    (x_train, y_train), (x_test,
            y_test)=keras.datasets.mnist.load_data()
    x_train=x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test=x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train/=255
    x_test/=255
    print('x train', x_train.shape)
    print('x test', x_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    trained=True
    if not trained:
        model = Sequential()
        initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        biasInitialization= keras.initializers.Constant(value=0.5)
    
        model.add(Conv2D(filters=8, kernel_size=(3,3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu', 
                  input_shape=(28, 28, 1)))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout=0.1)
        model.add(Flatten())
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
    
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
    
        figureIndex=0
        plt.figure(figureIndex)
        figureIndex+=1
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.ylim([0., 1.])

        plt.figure(figureIndex)
        figureIndex+=1
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        model.save('trained_cnn.h5')
        print('trained model saved')
    else:
        #del model
        model=keras.models.load_model('trained_cnn.h5')
        print('trained model loaded')
        model.summary()

        prob1_1=False
        if prob1_1:
            for i in range(3): 
                w=model.layers[i].get_weights()
                w=w[0]
                #w[1] for bias, w[0] for kernel
                print('w shape before squeeze', w.shape)
                w=np.squeeze(w)
                print('w shape ', w.shape)
                if i==0:
                    plt.figure(figsize=(2,4))
                    im=plt.imshow(make_mosaic(w, 2, 4, i), 
                          interpolation='nearest', cmap=cm.binary)
                else:
                    plt.figure(figsize=(8,8))
                    im=plt.imshow(make_mosaic(w, 8, 8, i), 
                          interpolation='nearest', cmap=cm.binary)

                plt.title('filters in layer '+str(i+1))
            plt.show()

        prob1_2=False
        if prob1_2:
            print(model.predict(x_train[1:2]))
            print(y_train[1])
            print(model.predict(x_train[17:18]))
            print(y_train[17])

            get_3rd_layer_output=K.function([model.layers[0].input],
                    [model.layers[2].output])

            output=get_3rd_layer_output([x_train[1:2]])[0]
            print('3rd layer output', output.shape )
            w=output[0]
            plt.figure(figsize=(2,4))
            im=plt.imshow(make_mosaic(w, 2, 4, 0), 
                          interpolation='nearest', cmap=cm.binary)
            plt.title('digit 0')

            output=get_3rd_layer_output([x_train[17:18]])[0]
            print('3rd layer output', output.shape )
            w=output[0]
            plt.figure(figsize=(2,4))
            im=plt.imshow(make_mosaic(w, 2, 4, 0), 
                          interpolation='nearest', cmap=cm.binary)
            plt.title('digit 8')
            plt.show()

        prob1_3=False
        if prob1_3:
            print(model.predict(x_train[3:4]))
            print(y_train[3:4])
            xLeft=np.copy(x_train[3:4])
            xRight=np.copy(x_train[3:4])
            print('xLeft shape', xLeft.shape) # 1,28, 28, 1
            xLeft[0, :, :26, 0]=xLeft[0, :, 2:28, 0]
            xLeft[0, :, 26, 0]=xLeft[0, :, 25, 0]
            xLeft[0, :, 27, 0]=xLeft[0, :, 25, 0]
            xRight[0, :, 2:28, 0]=xRight[0, :, 0:26, 0]
            xRight[0, :, 1, 0]=   xRight[0, :, 2, 0]
            xRight[0, :, 0, 0]=   xRight[0, :, 2, 0]

            plt.figure(0)
            plt.imshow(x_train[3:4][0,:,:,0], interpolation='nearest',
                    cmap=cm.binary)
            plt.title('Original digit 1')

            plt.figure(1)
            plt.imshow(xLeft[0,:,:,0], interpolation='nearest',
                    cmap=cm.binary)
            plt.title('Left shifted digit 1')
            print(model.predict(xLeft))

            plt.figure(2)
            plt.imshow(xRight[0,:,:,0], interpolation='nearest',
                    cmap=cm.binary)
            plt.title('Right shifted digit 1')
            print(model.predict(xRight))

            plt.show()

        prob2_1=False
        if prob2_1:
            print(model.predict(x_train[13:14]))
            print(x_train[13:14][0,:,:,0])
            plt.imshow(x_train[13:14][0,:,:,0], interpolation='nearest',
                    cmap=cm.binary)
            plt.title('digit 6')
            plt.show()
            print(y_train[13:14])
            prob=np.zeros((23, 23))
            maxProb=np.zeros((23,23))
            label=np.zeros((23,23))
            for i in range(23):
                for j in range(23):
                    temp=np.copy(x_train[13:14])
                    temp[0, i:i+6, j:j+6, 0]=1.
                    pred=model.predict(temp)
                    #print(pred)
                    prob[i,j]=pred[0, 6]
                    maxProb[i,j]=max(pred[0])
                    label[i,j]=np.argmax(pred[0])
                    print(i,j,'prob of 6', prob[i,j], 'max prob',
                            maxProb[i,j], 'label', label[i,j])
            plt.figure(0)
            plt.imshow(prob, interpolation='nearest',
                    cmap=cm.binary)
            plt.title('prob of classied as 6')

            plt.figure(1)
            plt.imshow(maxProb, interpolation='nearest',
                    cmap=cm.binary)
            plt.title('max prob')

            plt.figure(2)
            plt.imshow(label/9., interpolation='nearest',
                    cmap=cm.binary)
            plt.title('label')

            plt.show()
        
        prob2_3=True
        if prob2_3:
            #print(model.predict(x_train[:20]))
            print(y_train[:20])
            plt.figure(0)
            plt.imshow(x_train[13:14][0,:,:,0], interpolation='nearest',
                    cmap=cm.binary)
            plt.title('digit 6')

            plt.figure(2)
            plt.imshow(x_train[6:7][0,:,:,0], interpolation='nearest',
                    cmap=cm.binary)
            plt.title('digit 1')

            plt.figure(5)
            temp1=np.copy(x_train[6:7])
            temp6=np.copy(x_train[13:14])
            temp6[0,4:12, 15:23,0]=temp1[0, 9:17, 9:17, 0]
            print(model.predict(temp6))
            plt.imshow(temp6[0,:,:,0], interpolation='nearest',
                    cmap=cm.binary)
            plt.title('digit 6 with patch from 0')
            plt.show()

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def softmax(x):
    ex=np.exp(x)
    denominator=np.sum(ex)
    return ex/denominator

def forward(x, b, c, W, U, V):
    timeStep=3
    a=np.zeros((timeStep, 2))
    h=np.zeros((timeStep, 2))
    y=np.zeros((timeStep, 2))
    loss=0.
    for i in range(timeStep):
        if i==0:
            a[i]=b+U.dot(x[i])
        else:
            a[i]=b+W.dot(h[i-1])+U.dot(x[i])
        h[i]=tanh(a[i])
        o=c+V.dot(h[i])
        y[i]=softmax(o)
        loss+=(y[i][0]-0.5)*(y[i][0]-0.5)-np.log(y[i][1])
    return a, h, y, loss

def cal_gradient(x, y, h, a, b, c, W, U, V):
    print('y:', y)
    print('a:', a)
    dldb=np.zeros((3,2))
    dldy=np.zeros((3,2))
    dydo=np.zeros((3,2,2))
    dodh=np.zeros((2,2))
    dhda=np.zeros((3,2))
    dadb=np.zeros((3,2))
    dhdb=np.zeros((3,2))

    for i in range(3):
        print(i)
        dldy[i]=np.array([2.*y[i][0]-1., -1./y[i][1]])
        print('dldy',dldy[i])
        dydo[i]=np.array([[y[i][0]-y[i][0]*y[i][0], -y[i][0]*y[i][1]],
                          [-y[i][0]*y[i][1], y[i][1]-y[i][1]*y[i][1]] ])
        print('dydo',dydo[i])
        dodh=V
        print('dodh',dodh)
        dhda[i]=1.-tanh(a[i])*tanh(a[i])
        print('dhda:', dhda[i])
        dadb[i]=np.array([1.,1.])
        if i>0:
            dadb[i]+=W.dot(dhdb[i-1])
        print('dadb:', dadb[i])
        dhdb[i]=dhda[i]*dadb[i]
        print('dhdb:', dhdb[i])
        dldb[i]=dldy[i]*((dydo[i].dot(dodh)).dot(dhda[i]))*dadb[i]
        print('dydo.dot dodh:', dydo[i].dot(dodh))
        print('dldb:', dldb[i])
    return dldb, dldy, dydo, dodh, dhda, dadb#, dhdb

def problem3():
    b=np.array([-1, 1])
    c=np.array([0.5, -0.5])
    W=np.array([[1., -1], [0, 2]])
    U=np.array([[-1., 0], [1, -2]])
    V=np.array([[-2., 1], [-1, 0]])
    x=np.array([[1,0.],[0.5, 0.25],[0,1]])

    prob3_1=False
    if prob3_1:
        a_, h, y, loss=forward(x, b, c, W, U, V)
        print('y: ', y)
        print('loss: ', loss)

    prob3_2=False
    if prob3_2:
        eps=0.0001
        loss=np.zeros(2)
        temp=np.array([eps,0])
        a_, h_, y_, loss[0]=forward(x, b-temp, c, W, U, V)
        a_, h_, y_, loss[1]=forward(x, b+temp, c, W, U, V)
        grad=(loss[1]-loss[0])/2./eps
        print('loss(b[1]-eps):', loss[0])
        print('loss(b[1]+eps):', loss[1])
        print('gradient for b1:', grad)

        temp=np.array([0, eps])
        a_, h_, y_, loss[0]=forward(x, b-temp, c, W, U, V)
        a_, h_, y_, loss[1]=forward(x, b+temp, c, W, U, V)
        grad=(loss[1]-loss[0])/2./eps
        print('loss(b[2]-eps):', loss[0])
        print('loss(b[2]+eps):', loss[1])
        print('gradient for b2:', grad)

    prob3_3=True
    if prob3_3:
        a, h, y, loss=forward(x, b, c, W, U, V)
        #dldb, dldy, dydo, dodh, dhda, dadb, dhdb=cal_gradient(
        dldb, dldy, dydo, dodh, dhda, dadb=cal_gradient(
                                    x, y, h, a, b, c, W, U, V)
        #print('dldb', dldb)
        #print('dldy', dldy)
        #print('dydo', dydo)
        #print('dodh', dodh)
        #print('dhda', dhda)
        #print('dady', dadb)
        #print('dhdb', dhdb)
        
#cnn()
problem3()
