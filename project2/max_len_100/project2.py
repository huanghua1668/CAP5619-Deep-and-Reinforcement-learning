'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import tensorflow as tf
#from tensorflow import keras
import  keras
#from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Flatten, Conv2D, Average
from keras.layers import GlobalAveragePooling2D, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

def preprocess(seqs0, maxLen=100):
    originalSeqs="".join(seqs0)
    chars=list(set(originalSeqs))
    chars.sort() 
    # this is necessary, other wise each run the set will have different 
    # orders, hence encode/decoder might change for each run
    print('chars:', chars)
    charsSize=len(chars)+1 # 1 for padding
    print('chars size:', charsSize)

    charsEmbed=dict(zip(chars, range(1, charsSize)))
    charsDecode=dict(zip(range(1, charsSize), chars))

    #1-hot
    charsOneHot=np.identity(charsSize).astype(int)

    seqsX=np.zeros((len(seqs0), maxLen, charsSize))
    seqsY=np.copy(seqsX)

    for i in range(len(seqs0)):
        embed_x=[charsEmbed[v] for v in seqs0[i]]
        sz=len(embed_x)
        if i%1000==0: print('i',i,', sz', sz)
        if sz>maxLen+1:
            embed_x=embed_x[:maxLen+1]
        elif sz<maxLen+1:
            for j in range(maxLen+1-sz):
                embed_x.append(0)
        if i==0: 
            print('before embedding: ', seqs0[i])
            print('after embedding: ', embed_x)
        seqsX[i,:,:]=np.array([charsOneHot[j,:] for j in embed_x[:-1]])
        seqsY[i,:,:]=np.array([charsOneHot[j,:] for j in embed_x[1:]])

    print('seqsX looks like')
    for i in range(seqsX[0].shape[0]):
        print(seqsX[0][i])
    print('seqsY looks like')
    for i in range(seqsY[0].shape[0]):
        print(seqsY[0][i])
    return seqsX, seqsY, charsSize, charsDecode

def problem1():
    seqs0=[]
    with open('pdb_seqres.txt', 'r') as data:
        count=1
        maxLen=0
        minLen=10000
        for line in data:
            seq=line.strip()
            maxLen=max(maxLen, len(seq))
            minLen=min(minLen, len(seq))
            seqs0.append(seq)
            count+=1
            #if count==44011: break
            if count==4401: break
        print('loaded ', len(seqs0), 'sequences')
        print('max length of seq', maxLen)
        print('min length of seq', minLen)

    seqsX, seqsY, charsSize, charsDecode=preprocess(seqs0, maxLen=50)
    x_train=[]
    y_train=[]
    for i in range(seqsX.shape[0]):
        if (i+1)%5!=0:
            x_train.append(seqsX[i])
            y_train.append(seqsY[i])
    x_train=np.stack(x_train)
    y_train=np.stack(y_train)
    x_test=seqsX[4::5]
    y_test=seqsY[4::5]
    print('samples for train', x_train.shape)
    print('samples for test', x_test.shape)

    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
    biasInitialization= keras.initializers.Constant(value=1.)

    trained=False
    #trained=True
    if not trained:
        model=Sequential()
        model.add(LSTM(100, input_shape=(None, charsSize),
                       kernel_initializer=initialization, 
                       bias_initializer=biasInitialization,
                       return_sequences=True))
        model.add(LSTM(100, return_sequences=True,
                       kernel_initializer=initialization, 
                       bias_initializer=biasInitialization))
        model.add(TimeDistributed(Dense(charsSize)))
        model.add(Activation('softmax'))
        model.summary()

        adam=keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999,
                    epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss="categorical_crossentropy", 
                      optimizer=adam,
                      metrics=['accuracy'])

        history=model.fit(x_train, y_train, 
                          batch_size=64, verbose=1, epochs=200,
                          validation_data=(x_test, y_test))

        model.save('my_lstm.h5')

        figureIndex=0
        plt.figure(figureIndex)
        figureIndex+=1
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')

        plt.figure(figureIndex)
        figureIndex+=1
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    else:
        model=keras.models.load_model('my_lstm.h5')
        model.summary()

    # generate new sequences

def problem2():
    seqs0=[]
    with open('pdb_seqres.txt', 'r') as data:
        count=1
        maxLen=0
        minLen=10000
        for line in data:
            seq=line.strip()
            maxLen=max(maxLen, len(seq))
            minLen=min(minLen, len(seq))
            seqs0.append(seq)
            count+=1
            #if count==44011: break
            if count==4401: break
        print('loaded ', len(seqs0), 'sequences')
        print('max length of seq', maxLen)
        print('min length of seq', minLen)

    seqsX, seqsY, charsSize, charsDecode=preprocess(seqs0, maxLen=100)
    x_train=[]
    y_train=[]
    for i in range(seqsX.shape[0]):
        if (i+1)%5!=0:
            x_train.append(seqsX[i])
            y_train.append(seqsY[i])
    x_train=np.stack(x_train)
    y_train=np.stack(y_train)
    x_test=seqsX[4::5]
    y_test=seqsY[4::5]
    print('samples for train', x_train.shape)
    print('samples for test', x_test.shape)

    model=keras.models.load_model('my_lstm.h5')
    model.summary()
    for i in range(5):
        print(i)
        #samples=np.copy(x_test[i:i+1])
        samples=np.copy(x_train[i:i+1])
        predictOld=model.predict_classes(samples, batch_size=None, verbose=0)
        predictOld=predictOld[0]
        sqesOld=''
        for i in range(predictOld.shape[0]):
            sqesOld+=charsDecode[predictOld[i]]
        print('old predict', sqesOld)
        category=np.where(samples[0][0]>0)
        index=category[0][0]
        if index<19:
            index+=1
            samples[0][0][index-1]=0
            samples[0][0][index]=1
        else:
            index-=1
            samples[0][0][index+1]=0
            samples[0][0][index]=1
        predict=model.predict_classes(samples, batch_size=None, verbose=0)
        predict=predict[0]
        sqes=''
        for i in range(predict.shape[0]):
            if predict[i]!=0:
                sqes+=charsDecode[predict[i]]
            else:
                break
        print('new predict', sqes)
        #for i in range(50):

def genSeqs(model, seqs0, seqsLen, charsSize, charsDecode):
    # seqs0 is the first char's encode
    chars=np.zeros((1, seqsLen, charsSize))
    chars[0, 0][seqs0]=1
    seqs=[charsDecode[seqs0]]
    for i in range(0, seqsLen):
        pred=model.predict_classes(chars[:, :i+1, :],
            batch_size=None, verbose=0)[0][-1]
        seqs.append(charsDecode[pred])
    return ('').join(seqs)

def problem3():
    seqs0=[]
    with open('pdb_seqres.txt', 'r') as data:
        count=1
        maxLen=0
        minLen=10000
        for line in data:
            seq=line.strip()
            maxLen=max(maxLen, len(seq))
            minLen=min(minLen, len(seq))
            seqs0.append(seq)
            count+=1
            #if count==44011: break
            if count==4401: break
        print('loaded ', len(seqs0), 'sequences')
        print('max length of seq', maxLen)
        print('min length of seq', minLen)

    seqsX, seqsY, charsSize, charsDecode=preprocess(seqs0, maxLen=50)
    x_train=[]
    y_train=[]
    for i in range(seqsX.shape[0]):
        if (i+1)%5!=0:
            x_train.append(seqsX[i])
            y_train.append(seqsY[i])
    x_train=np.stack(x_train)
    y_train=np.stack(y_train)
    x_test=seqsX[4::5]
    y_test=seqsY[4::5]
    print('samples for train', x_train.shape)
    print('samples for test', x_test.shape)

    model=keras.models.load_model('my_lstm.h5')
    model.summary()
    samples=np.copy(x_train[0:5])
    seq0=[]
    for i in range(samples.shape[0]):
        chars=''
        for j in range(5):
            seqs=np.argmax(samples[i,j])
            chars+=charsDecode(seqs)
        seq0.append(chars)
    inputs=np.argmax(samples[:, 0], axis=1)
    print('inputs', inputs)
    print('input seqs', seq0)
    seq1=[]
    for i in range(inputs.shape[0]):
        seqs=genSeqs(model, inputs[i], 5, charsSize, charsDecode)
        seq1.append(seqs)
    print('generated seqs', seq1)

def mlp():
    batch_size = 256
    #batch_size = 4096*2
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    y_test=test[:,0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    initialization0= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    #initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    #initialization= keras.initializers.Orthogonal(gain=1.0)
    initialization= keras.initializers.Identity(gain=3.)
    biasInitialization= keras.initializers.Constant(value=0.5)
    #regularizer=keras.regularizers.l1(0.05)
    regularizer=keras.regularizers.l1(0.15)
    #regularizer=keras.regularizers.l1(0.2)
    model.add(Dense(256, kernel_initializer=initialization, 
                         bias_initializer=biasInitialization,
                         kernel_regularizer=regularizer,
                         activation='tanh',
                         input_shape=(256,)))
    #model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer=initialization, 
                         bias_initializer=biasInitialization,
                         kernel_regularizer=regularizer,
                         activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer=initialization, 
                         bias_initializer=biasInitialization,
                         kernel_regularizer=regularizer,
                         activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes,kernel_initializer=initialization0,  
                         activation='softmax'))
    model.summary()

    #sgd=keras.optimizers.SGD(lr=0.25, momentum=0.0, decay=0.0,
    sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
    #sgd=keras.optimizers.SGD(lr=0.01, momentum=0.99, decay=0.0,
    #sgd=keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0,
    #sgd=keras.optimizers.SGD(lr=0.00001, momentum=0.0, decay=0.0,
            nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
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

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    score=model.evaluate(x_test, y_test, verbose=0)
                                                                               
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def mlp_initialization():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    y_test=test[:,0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    gains=[0.1, 3, 10]
    histories=[]
    for i in range(len(gains)):
        gain=gains[i]
        model = Sequential()
        initialization0= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
        initialization= keras.initializers.Identity(gain=gain)
        biasInitialization= keras.initializers.Constant(value=0.5)
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='tanh',
                             input_shape=(256,)))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(num_classes,kernel_initializer=initialization0,  
                             bias_initializer=biasInitialization,
                             activation='softmax'))
        model.summary()

        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='gains='+str(gains[0]))
    plt.plot(histories[1].history['acc'], label='gains='+str(gains[1]))
    plt.plot(histories[2].history['acc'], label='gains='+str(gains[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='gains='+str(gains[0]))
    plt.plot(histories[1].history['loss'], label='gains='+str(gains[1]))
    plt.plot(histories[2].history['loss'], label='gains='+str(gains[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    score=model.evaluate(x_test, y_test, verbose=0)
                                                                               
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def mlp_learning_rate():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    y_test=test[:,0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    initialization0= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    initialization= keras.initializers.Identity(gain=3.)
    biasInitialization= keras.initializers.Constant(value=0.5)

    lams=[0.25, 0.01, 0.00001]
    histories=[]
    for i in range(len(lams)):
        lam=lams[i]
        model = Sequential()
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='tanh',
                             input_shape=(256,)))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(num_classes,kernel_initializer=initialization0,  
                             bias_initializer=biasInitialization,
                             activation='softmax'))
        model.summary()

        sgd=keras.optimizers.SGD(lr=lam, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label=r'$\alpha$='+str(lams[0]))
    plt.plot(histories[1].history['acc'], label=r'$\alpha$='+str(lams[1]))
    plt.plot(histories[2].history['acc'], label=r'$\alpha$='+str(lams[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label=r'$\alpha$='+str(lams[0]))
    plt.plot(histories[1].history['loss'], label=r'$\alpha$='+str(lams[1]))
    plt.plot(histories[2].history['loss'], label=r'$\alpha$='+str(lams[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def mlp_batch_size():
    #batch_size= 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    y_test=test[:,0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    batch_sizes=[256, 4096*2]
    histories=[]
    for i in range(len(batch_sizes)):
        batch_size=batch_sizes[i]
        model = Sequential()
        initialization0= keras.initializers.TruncatedNormal(mean=0., 
                                 stddev=0.05)
        initialization= keras.initializers.Identity(gain=3)
        biasInitialization= keras.initializers.Constant(value=0.5)
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='tanh',
                             input_shape=(256,)))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(num_classes,kernel_initializer=initialization0,  
                             bias_initializer=biasInitialization,
                             activation='softmax'))
        model.summary()

        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='batch_size='+str(batch_sizes[0]))
    plt.plot(histories[1].history['acc'], label='batch_size='+str(batch_sizes[1]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='batch_size='+str(batch_sizes[0]) )
    plt.plot(histories[1].history['loss'], label='batch_size='+str(batch_sizes[1]) )
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def mlp_momentum():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    y_test=test[:,0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    initialization0= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    initialization= keras.initializers.Identity(gain=3.)
    biasInitialization= keras.initializers.Constant(value=0.5)

    momentums=[0.5, 0.9, 0.99]
    histories=[]
    for i in range(len(momentums)):
        model = Sequential()
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='tanh',
                             input_shape=(256,)))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dense(num_classes,kernel_initializer=initialization0,  
                             bias_initializer=biasInitialization,
                             activation='softmax'))
        model.summary()

        sgd=keras.optimizers.SGD(lr=0.01, momentum=momentums[i], decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='momentum='+str(momentums[0]))
    plt.plot(histories[1].history['acc'], label='momentum='+str(momentums[1]))
    plt.plot(histories[2].history['acc'], label='momentum='+str(momentums[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='momentum='+str(momentums[0]))
    plt.plot(histories[1].history['loss'], label='momentum='+str(momentums[1]))
    plt.plot(histories[2].history['loss'], label='momentum='+str(momentums[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def mlp_dropout():
    batch_size = 256
    #batch_size = 4096*2
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    y_test=test[:,0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    dropouts=[0.0, 0.2, 0.5]
    histories=[]
    initialization0= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    initialization= keras.initializers.Identity(gain=3.)
    biasInitialization= keras.initializers.Constant(value=0.5)
    for i in range(len(dropouts)):
        model = Sequential()
        #regularizer=keras.regularizers.l1()
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             #kernel_regularizer=regularizer,
                             activation='tanh',
                             input_shape=(256,)))
        model.add(Dropout(dropouts[i]))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dropout(dropouts[i]))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             activation='relu'))
        model.add(Dropout(dropouts[i]))
        model.add(Dense(num_classes,kernel_initializer=initialization0,  
                             bias_initializer=biasInitialization,
                             activation='softmax'))
        model.summary()

        #sgd=keras.optimizers.SGD(lr=0.25, momentum=0.0, decay=0.0,
        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
        #sgd=keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0,
        #sgd=keras.optimizers.SGD(lr=0.00001, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'],
            label='train_dropout'+str(dropouts[0]))
    plt.plot(histories[0].history['val_acc'],
            label='test_dropout'+str(dropouts[0]))
    plt.plot(histories[1].history['acc'],
            label='train_dropout'+str(dropouts[1]))
    plt.plot(histories[1].history['val_acc'],
            label='test_dropout'+str(dropouts[1]))
    plt.plot(histories[2].history['acc'],
            label='train_dropout'+str(dropouts[2]))
    plt.plot(histories[2].history['val_acc'],
            label='test_dropout'+str(dropouts[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

def mlp_regularization():
    batch_size = 256
    #batch_size = 4096*2
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    y_test=test[:,0]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    regularizations=[0.0, 0.1, 0.2]
    histories=[]
    initialization0= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    initialization= keras.initializers.Identity(gain=3.)
    biasInitialization= keras.initializers.Constant(value=0.5)
    for i in range(len(regularizations)):
        regularizer=keras.regularizers.l1(regularizations[i])
        model = Sequential()
        #regularizer=keras.regularizers.l1()
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             kernel_regularizer=regularizer,
                             activation='tanh',
                             input_shape=(256,)))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             kernel_regularizer=regularizer,
                             activation='relu'))
        model.add(Dense(256, kernel_initializer=initialization, 
                             bias_initializer=biasInitialization,
                             kernel_regularizer=regularizer,
                             activation='relu'))
        model.add(Dense(num_classes,kernel_initializer=initialization0,  
                             bias_initializer=biasInitialization,
                             activation='softmax'))
        model.summary()

        #sgd=keras.optimizers.SGD(lr=0.25, momentum=0.0, decay=0.0,
        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
        #sgd=keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0,
        #sgd=keras.optimizers.SGD(lr=0.00001, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'],
            label='train_L1('+str(regularizations[0])+')')
    plt.plot(histories[0].history['val_acc'],
            label='test_L1('+str(regularizations[0])+')')
    plt.plot(histories[1].history['acc'],
            label='train_L1('+str(regularizations[1])+')')
    plt.plot(histories[1].history['val_acc'],
            label='test_L1('+str(regularizations[1])+')')
    plt.plot(histories[2].history['acc'],
            label='train_L1('+str(regularizations[2])+')')
    plt.plot(histories[2].history['val_acc'],
            label='test_L1('+str(regularizations[2])+')')
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

def cnn():
    batch_size = 128
    #batch_size = 4096
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    #initialization= keras.initializers.TruncatedNormal(mean=0., stddev=1.0)
    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
    #initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.02)
    biasInitialization= keras.initializers.Constant(value=0.5)
    regularizer=keras.regularizers.l1(0.01)

    model.add(Conv2D(filters=8, kernel_size=(3,3), 
              kernel_initializer=initialization, 
              bias_initializer=biasInitialization, 
              kernel_regularizer=regularizer,
              activation='sigmoid', 
              input_shape=(16, 16, 1)))
    model.add(Conv2D(8, (3, 3), 
              kernel_initializer=initialization, 
              bias_initializer=biasInitialization, 
              kernel_regularizer=regularizer,
              activation='relu'))
    model.add(Conv2D(8, (3, 3), 
              kernel_initializer=initialization, 
              bias_initializer=biasInitialization, 
              kernel_regularizer=regularizer,
              activation='relu'))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # lr: learning rate
    # decay: leaning rate decay over each update  
    sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
    #sgd=keras.optimizers.SGD(lr=0.01, momentum=0.99, decay=0.0,
    #sgd=keras.optimizers.SGD(lr=0.00001, momentum=0.0, decay=0.0,
            nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
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
                                                        
def cnn_initialization():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    stddevs=[1.0, 0.5, 0.02]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    for i in range(len(stddevs)):
        model = Sequential()
        initialization= keras.initializers.TruncatedNormal(mean=0., 
                                             stddev=stddevs[i])
        model.add(Conv2D(filters=8, kernel_size=(3,3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='sigmoid', 
                  input_shape=(16, 16, 1)))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Flatten())
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='stddev='+str(stddevs[0]))
    plt.plot(histories[1].history['acc'], label='stddev='+str(stddevs[1]))
    plt.plot(histories[2].history['acc'], label='stddev='+str(stddevs[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='stddev='+str(stddevs[0]))
    plt.plot(histories[1].history['loss'], label='stddev='+str(stddevs[1]))
    plt.plot(histories[2].history['loss'], label='stddev='+str(stddevs[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def cnn_learning_rate():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    lams=[0.1, 0.01, 0.00001]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., 
                                             stddev=0.5)
    for i in range(len(lams)):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(3,3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='sigmoid', 
                  input_shape=(16, 16, 1)))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Flatten())
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=lams[i], momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label=r'$\alpha$='+str(lams[0]))
    plt.plot(histories[1].history['acc'], label=r'$\alpha$='+str(lams[1]))
    plt.plot(histories[2].history['acc'], label=r'$\alpha$='+str(lams[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label=r'$\alpha$='+str(lams[0]))
    plt.plot(histories[1].history['loss'], label=r'$\alpha$='+str(lams[1]))
    plt.plot(histories[2].history['loss'], label=r'$\alpha$='+str(lams[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def cnn_batch_size():
    #batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    batch_sizes=[128, 4096]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., 
                                             stddev=0.5)
    for i in range(len(batch_sizes)):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(3,3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='sigmoid', 
                  input_shape=(16, 16, 1)))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Flatten())
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_sizes[i],
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='batch_size='+str(batch_sizes[0]))
    plt.plot(histories[1].history['acc'], label='batch_size='+str(batch_sizes[1]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='batch_size='+str(batch_sizes[0]))
    plt.plot(histories[1].history['loss'], label='batch_size='+str(batch_sizes[1]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def cnn_momentum():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    momentums=[0.5, 0.9, 0.99]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., 
                                             stddev=0.5)
    for i in range(len(momentums)):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(3,3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='sigmoid', 
                  input_shape=(16, 16, 1)))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Flatten())
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.01, momentum=momentums[i], decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='momentum='+str(momentums[0]))
    plt.plot(histories[1].history['acc'], label='momentum='+str(momentums[1]))
    plt.plot(histories[2].history['acc'], label='momentum='+str(momentums[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='momentum='+str(momentums[0]))
    plt.plot(histories[1].history['loss'], label='momentum='+str(momentums[1]))
    plt.plot(histories[2].history['loss'], label='momentum='+str(momentums[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def cnn_dropout():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    dropouts=[0.0, 0.2, 0.6]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., 
                                             stddev=0.25)
    for i in range(len(dropouts)):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(3,3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='sigmoid', 
                  input_shape=(16, 16, 1)))
        model.add(Dropout(dropouts[i]))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Dropout(dropouts[i]))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  activation='relu'))
        model.add(Dropout(dropouts[i]))
        model.add(Flatten())
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'],
            label='train_dropout'+str(dropouts[0]))
    plt.plot(histories[0].history['val_acc'],
            label='test_dropout'+str(dropouts[0]))
    plt.plot(histories[1].history['acc'],
            label='train_dropout'+str(dropouts[1]))
    plt.plot(histories[1].history['val_acc'],
            label='test_dropout'+str(dropouts[1]))
    plt.plot(histories[2].history['acc'],
            label='train_dropout'+str(dropouts[2]))
    plt.plot(histories[2].history['val_acc'],
            label='test_dropout'+str(dropouts[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
                                                        
def cnn_regularization():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    regularizations=[0.0, 0.01, 0.02]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., 
                                             stddev=0.25)
    for i in range(len(regularizations)):
        regularizer=keras.regularizers.l1(regularizations[i])
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(3,3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  kernel_regularizer=regularizer,
                  activation='sigmoid', 
                  input_shape=(16, 16, 1)))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  kernel_regularizer=regularizer,
                  activation='relu'))
        model.add(Conv2D(8, (3, 3), 
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization, 
                  kernel_regularizer=regularizer,
                  activation='relu'))
        model.add(Flatten())
        #model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'],
            label='train_L1('+str(regularizations[0])+')')
    plt.plot(histories[0].history['val_acc'],
            label='test_L1('+str(regularizations[0])+')')
    plt.plot(histories[1].history['acc'],
            label='train_L1('+str(regularizations[1])+')')
    plt.plot(histories[1].history['val_acc'],
            label='test_L1('+str(regularizations[1])+')')
    plt.plot(histories[2].history['acc'],
            label='train_L1('+str(regularizations[2])+')')
    plt.plot(histories[2].history['val_acc'],
            label='test_L1('+str(regularizations[2])+')')
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.ylim([0., 1.])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

def locally_connected():
    batch_size = 128
    #batch_size = 2048
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    #initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.2)
    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
    #initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    #initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
    #initialization= keras.initializers.Constant(gain=3.)
    biasInitialization= keras.initializers.Constant(value=0.5)
    #regularizer=keras.regularizers.l1(0.01)
    #regularizer=keras.regularizers.l1(0.005)
    #regularizer=keras.regularizers.l1(0.001)
    #regularizer=keras.regularizers.l1(0.002)
    regularizer=keras.regularizers.l1(0.0005)

    model = Sequential()
    model.add(keras.layers.LocallyConnected2D(8, (3,3),  
              kernel_initializer=initialization, 
              bias_initializer=biasInitialization,
              kernel_regularizer=regularizer,
              input_shape=(16, 16, 1),
              activation='sigmoid'))
    #model.add(Dropout(0.6))
    model.add(keras.layers.LocallyConnected2D(8, (3,3),
              kernel_initializer=initialization, 
              bias_initializer=biasInitialization,
              kernel_regularizer=regularizer,
              activation='relu'))
    model.add(keras.layers.LocallyConnected2D(8, (3,3),
              kernel_initializer=initialization, 
              bias_initializer=biasInitialization,
              kernel_regularizer=regularizer,
              activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',
              kernel_initializer=initialization, 
              bias_initializer=biasInitialization))
    model.summary()

    # lr: learning rate
    # decay: leaning rate decay over each update  
    sgd=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0,
    #sgd=keras.optimizers.SGD(lr=0.1, momentum=0.99, decay=0.0,
    #sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0,
            nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
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

def locally_connected_initialization():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    stddevs=[0.15, 0.1, 0.05]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)

    for i in range(len(stddevs)):
        initialization= keras.initializers.TruncatedNormal(mean=0., stddev=stddevs[i])

        model = Sequential()
        model.add(keras.layers.LocallyConnected2D(8, (3,3),  
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  input_shape=(16, 16, 1),
                  activation='sigmoid'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='stddev='+str(stddevs[0]))
    plt.plot(histories[1].history['acc'], label='stddev='+str(stddevs[1]))
    plt.plot(histories[2].history['acc'], label='stddev='+str(stddevs[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='stddev='+str(stddevs[0]))
    plt.plot(histories[1].history['loss'], label='stddev='+str(stddevs[1]))
    plt.plot(histories[2].history['loss'], label='stddev='+str(stddevs[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def locally_learning_rate():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    lams=[0.5, 0.1, 0.01]
    histories=[]

    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
    biasInitialization= keras.initializers.Constant(value=0.5)
    for i in range(len(lams)):
        model = Sequential()
        model.add(keras.layers.LocallyConnected2D(8, (3,3),  
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  input_shape=(16, 16, 1),
                  activation='sigmoid'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=lams[i], momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label=r'$\alpha$='+str(lams[0]))
    plt.plot(histories[1].history['acc'], label=r'$\alpha$='+str(lams[1]))
    plt.plot(histories[2].history['acc'], label=r'$\alpha$='+str(lams[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label=r'$\alpha$='+str(lams[0]))
    plt.plot(histories[1].history['loss'], label=r'$\alpha$='+str(lams[1]))
    plt.plot(histories[2].history['loss'], label=r'$\alpha$='+str(lams[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def locally_batch_size():
    #batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    batch_sizes=[128, 2048]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)

    for i in range(len(batch_sizes)):
        model = Sequential()
        model.add(keras.layers.LocallyConnected2D(8, (3,3),  
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  input_shape=(16, 16, 1),
                  activation='sigmoid'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_sizes[i],
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='batch_size='+str(batch_sizes[0]))
    plt.plot(histories[1].history['acc'], label='batch_size='+str(batch_sizes[1]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='batch_size='+str(batch_sizes[0]))
    plt.plot(histories[1].history['loss'], label='batch_size='+str(batch_sizes[1]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def locally_momentum():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    momentums=[0.5, 0.9, 0.99]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)

    for i in range(len(momentums)):
        model = Sequential()
        model.add(keras.layers.LocallyConnected2D(8, (3,3),  
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  input_shape=(16, 16, 1),
                  activation='sigmoid'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.1, momentum=momentums[i], decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'], label='momentum='+str(momentums[0]))
    plt.plot(histories[1].history['acc'], label='momentum='+str(momentums[1]))
    plt.plot(histories[2].history['acc'], label='momentum='+str(momentums[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')

    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['loss'], label='momentum='+str(momentums[0]))
    plt.plot(histories[1].history['loss'], label='momentum='+str(momentums[1]))
    plt.plot(histories[2].history['loss'], label='momentum='+str(momentums[2]))
    #plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def locally_dropout():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    dropouts=[0., 0.2, 0.6]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)

    for i in range(len(dropouts)):
        model = Sequential()
        model.add(keras.layers.LocallyConnected2D(8, (3,3),  
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  input_shape=(16, 16, 1),
                  activation='sigmoid'))
        model.add(Dropout(dropouts[i]))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(Dropout(dropouts[i]))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  activation='relu'))
        model.add(Dropout(dropouts[i]))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        

    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'],
            label='train_dropout'+str(dropouts[0]))
    plt.plot(histories[0].history['val_acc'],
            label='test_dropout'+str(dropouts[0]))
    plt.plot(histories[1].history['acc'],
            label='train_dropout'+str(dropouts[1]))
    plt.plot(histories[1].history['val_acc'],
            label='test_dropout'+str(dropouts[1]))
    plt.plot(histories[2].history['acc'],
            label='train_dropout'+str(dropouts[2]))
    plt.plot(histories[2].history['val_acc'],
            label='test_dropout'+str(dropouts[2]))
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy in train')
    plt.show()

def locally_regularization():
    batch_size = 128
    num_classes = 10
    epochs = 50
    train=np.loadtxt('zip_train.txt')
    x_train=train[:,1:]
    x_train=x_train.reshape(x_train.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test=test[:,1:]
    x_test=x_test.reshape(x_test.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    regularizations=[0., 0.0005, 0.002]
    histories=[]
    biasInitialization= keras.initializers.Constant(value=0.5)
    initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.1)

    for i in range(len(regularizations)):
        regularizer=keras.regularizers.l1(regularizations[i])
        model = Sequential()
        model.add(keras.layers.LocallyConnected2D(8, (3,3),  
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  kernel_regularizer=regularizer,
                  input_shape=(16, 16, 1),
                  activation='sigmoid'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  kernel_regularizer=regularizer,
                  activation='relu'))
        model.add(keras.layers.LocallyConnected2D(8, (3,3),
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization,
                  kernel_regularizer=regularizer,
                  activation='relu'))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',
                  kernel_initializer=initialization, 
                  bias_initializer=biasInitialization))
        model.summary()

        # lr: learning rate
        # decay: leaning rate decay over each update  
        sgd=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0,
                nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,                                         
                            validation_data=(x_test, y_test))
        histories.append(history)
                                                        
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(histories[0].history['acc'],
            label='train_L1('+str(regularizations[0])+')')
    plt.plot(histories[0].history['val_acc'],
            label='test_L1('+str(regularizations[0])+')')
    plt.plot(histories[1].history['acc'],
            label='train_L1('+str(regularizations[1])+')')
    plt.plot(histories[1].history['val_acc'],
            label='test_L1('+str(regularizations[1])+')')
    plt.plot(histories[2].history['acc'],
            label='train_L1('+str(regularizations[2])+')')
    plt.plot(histories[2].history['val_acc'],
            label='test_L1('+str(regularizations[2])+')')
    #plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.ylim([0., 1.])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

class ensemble_class:
    def __init__(self, x_train, y_train, x_test, y_test, y_test0, epochs=50):
        self.epochs = epochs
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.y_test0=y_test0
        self.batch_size=128
        self.num_classes=10

    def cnn(self, model_input, seed):
        biasInitialization= keras.initializers.Constant(value=0.5)
        initialization= keras.initializers.TruncatedNormal(mean=0., stddev=0.2,
                seed=seed)
        x=Conv2D(filters=8, kernel_size=(3,3), 
                 kernel_initializer=initialization, 
                 bias_initializer=biasInitialization, 
                 activation='sigmoid')(model_input)
        x=Conv2D(8, (3, 3), 
                 kernel_initializer=initialization, 
                 bias_initializer=biasInitialization, 
                 activation='relu')(x)
        x=Conv2D(8, (3, 3), 
                 kernel_initializer=initialization, 
                 bias_initializer=biasInitialization, 
                 activation='relu')(x)
        if 1==0:
            x=Conv2D(10, (1,1))(x)
            x=GlobalAveragePooling2D()(x)
            x=Activation(activation='softmax')(x)
        else:
            x=Flatten()(x)
            x=Dense(self.num_classes, activation='softmax')(x)

        model=Model(model_input, x, name='conv_cnn')
        return model

    def mlp(self, model_input, seed):
        initialization= keras.initializers.TruncatedNormal(mean=0., 
                stddev=0.05, seed=seed)
        biasInitialization= keras.initializers.Constant(value=0.5)
        x=Dense(256, kernel_initializer=initialization, 
                   bias_initializer=biasInitialization,
                   activation='tanh')(model_input)
        x=Dense(256, kernel_initializer=initialization, 
                   bias_initializer=biasInitialization,
                   activation='relu')(x)
        x=Dense(256, kernel_initializer=initialization, 
                   bias_initializer=biasInitialization,
                   activation='relu')(x)
        x=Dense(self.num_classes, kernel_initializer=initialization,  
                             bias_initializer=biasInitialization,
                             activation='softmax')(x)
        model=Model(model_input, x, name='mlp')
        return model

    def compile_and_train(self, model):
        sgd=keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0,
            nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        history = model.fit(self.x_train, self.y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,                                         
                            validation_data=(self.x_test, self.y_test))
        return history

    def ensemble_model(self, models, model_input):
        outputs=[model.outputs[0] for model in models]
        y=Average()(outputs)
        model=Model(model_input, y, name='ensemble')
        return model

    def evaluate_accuracy(self, model):
        pred=model.predict(self.x_test, batch_size=self.batch_size)
        pred=np.argmax(pred, axis=1)
        pred=np.expand_dims(pred, axis=1)
        error=np.sum(np.not_equal(pred, self.y_test0))/self.y_test0.shape[0]
        print('sum',np.sum(np.not_equal(pred, self.y_test0)), 'error', error)
        return 1.-error

def ensemble_nn():
    num_classes = 10
    train=np.loadtxt('zip_train.txt')
    x_train0=train[:,1:]
    x_train=x_train0.reshape(x_train0.shape[0], 16, 16, 1)
    print('x train', x_train.shape)
    y_train=train[:,0]
    test=np.loadtxt('zip_test.txt')
    x_test0=test[:,1:]
    x_test=x_test0.reshape(x_test0.shape[0], 16, 16, 1)
    print('x test', x_test.shape)
    y_test=test[:,0]
    y_test0=test[:,0]
    y_test0=np.reshape(y_test0, (y_test0.shape[0],1))
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    ensemble=ensemble_class(x_train, y_train, x_test, y_test, y_test0, epochs=50)
    models=[]
    input_shape=x_train[0, :,:,:].shape
    model_input=Input(shape=input_shape)
    for i in range(6):
        np.random.seed()
        seed=np.random.randint(1, 1000000)
        model=ensemble.cnn(model_input, seed)
        _=ensemble.compile_and_train(model)
        accuracy=ensemble.evaluate_accuracy(model)
        print('Test accuracy for',i+1,'th cnn:', accuracy)
        models.append(model)
    ensemble_model=ensemble.ensemble_model(models, model_input)
    accuracy=ensemble.evaluate_accuracy(ensemble_model)
    print('Test accuracy for ensemble:', accuracy)

#problem1()
problem2()
