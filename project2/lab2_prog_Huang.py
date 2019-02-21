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
import csv

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
    samples=np.copy(x_test[0:1])
    predictOld=model.predict_classes(samples, batch_size=None, verbose=0)
    predictOld=predictOld[0]
    seqs0=''
    for i in range(predictOld.shape[0]):
        if predictOld[i]!=0:
            seqs0+=charsDecode[predictOld[i]]
        else:
            break
    print('old predict', seqs0)
    #print(samples[0][0])
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
    seqs=''
    for i in range(predict.shape[0]):
        if predict[i]!=0:
            seqs+=charsDecode[predict[i]]
        else:
            break
    print('new predict', seqs)
    #for i in range(50):

def genSeqs(model, seqs0, seqsLen, charsSize, charsDecodei):
    # seqs0 is the beginning chars's encode array
    chars=np.zeros((1, seqsLen, charsSize))
    seqs=[]
    for i in range(seqs0.shape[0]):
        chars[0, i][seqs0[i]]=1
    for i in range(seqs0.shape[0]):
        if seqs0[i]!=0:
            seqs.append(charsDecode[seqs0[i]])
        else:
            break
    for i in range(seqs0.shape[0], seqsLen):
        pred=model.predict_classes(chars[:, :i, :],
            batch_size=None, verbose=0)[0][-1]
        chars[0, i][pred]=1
        if pred!=0:
            seqs.append(charsDecode[pred])
        else:
            break
    return ('').join(seqs)

def problem3a():
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
    samples=np.copy(x_train[::500])
    seq0=[]
    l0=10
    l1=20
    inputs=np.zeros((samples.shape[0], l0))
    for i in range(samples.shape[0]):
        chars=''
        for j in range(l1):
            seqs=np.argmax(samples[i,j])
            if seqs!=0:
                chars+=charsDecode[seqs]
            else:
                break
        seq0.append(chars)
    print('input seqs', seq0)
    for i in range(samples.shape[0]):
        inputs[i]=np.argmax(samples[i, :l0, :], axis=1)

    seq1=[]
    for i in range(samples.shape[0]):
        seqs=genSeqs(model, inputs[i].astype(int), l1, charsSize, charsDecode)
        seq1.append(seqs)
    print('generated seqs', seq1)

def problem3b():
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
    samples=np.copy(x_test)
    seq0=[]
    ks=np.arange(19)+1
    cols=20
    l1=ks.shape[0]+cols-1
    origin=np.zeros((samples.shape[0], 50))
    for i in range(samples.shape[0]):
        origin[i]=np.argmax(samples[i,:,:], axis=1)

    table=np.zeros((ks.shape[0]+1, cols))
    f=open('table.csv', 'w')
    writer=csv.writer(f)

    for k in range(ks.shape[0]):
        l0=int(ks[k])
        inputs=np.zeros((samples.shape[0], l0))
        for j in range(samples.shape[0]):
            inputs[j]=np.argmax(samples[j, :l0, :], axis=1)
            seqs=genSeqs(model, inputs[j].astype(int), l1, 
                         charsSize, charsDecode)
            # make comparison
            count=0
            for i in range(l0, min(l0+cols-1, len(seqs))):
                if origin[j, i]==0: # end of sequence
                    break
                else:
                    if seqs[i]!=charsDecode[origin[j][i]]:
                        break
                    else:
                        count+=1
            if j%100==0:
                print('k=', l0, ', ', j, 'th sample, count=', count)
            table[l0, count]+=1
        print(table[l0])
        writer.writerow(table[l0])
    f.close()

def problem3c():
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

    table=np.zeros((charsSize, charsSize, charsSize, charsSize,
                    charsSize))


    chars=np.zeros((1, seqsLen, charsSize))
    seqs=[]
    for i in range(seqs0.shape[0]):
        chars[0, i][seqs0[i]]=1
    for i in range(seqs0.shape[0]):
        if seqs0[i]!=0:
            seqs.append(charsDecode[seqs0[i]])
        else:
            break
    for i in range(seqs0.shape[0], seqsLen):
        pred=model.predict_classes(chars[:, :i, :],
            batch_size=None, verbose=0)[0][-1]
        chars[0, i][pred]=1
        if pred!=0:
            seqs.append(charsDecode[pred])
        else:
            break

    for i in range(1,6):
        chars=np.zeros((1, seqsLen, charsSize))
        prob=np.zeros(charsSize)

        l0=i
        inputs=np.zeros((samples.shape[0], l0))
        input
    samples=np.copy(x_train)
    seq0=[]
    ks=np.arange(19)+1
    cols=20
    l1=ks.shape[0]+cols-1
    origin=np.zeros((samples.shape[0], 50))
    for i in range(samples.shape[0]):
        origin[i]=np.argmax(samples[i,:,:], axis=1)

    f=open('table.csv', 'w')
    writer=csv.writer(f)

    for k in range(ks.shape[0]):
        l0=int(ks[k])
        inputs=np.zeros((samples.shape[0], l0))
        for j in range(samples.shape[0]):
            inputs[j]=np.argmax(samples[j, :l0, :], axis=1)
            seqs=genSeqs(model, inputs[j].astype(int), l1, 
                         charsSize, charsDecode)
            # make comparison
            count=0
            for i in range(l0, min(l0+cols-1, len(seqs))):
                if origin[j, i]==0: # end of sequence
                    break
                else:
                    if seqs[i]!=charsDecode[origin[j][i]]:
                        break
                    else:
                        count+=1
            if j%100==0:
                print('k=', l0, ', ', j, 'th sample, count=', count)
            table[l0, count]+=1
        print(table[l0])
        writer.writerow(table[l0])
    f.close()

#problem1()
problem2()
#problem3a()
#problem3b()
