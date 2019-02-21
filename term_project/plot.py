import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('results.dat')
frames=data[:,1]
rewards=data[:,2]
stds=data[:,3]

data1=np.loadtxt('results_reduce_epsilon.dat')
frames1=data1[:,1]
rewards1=data1[:,2]
stds1=data1[:,3]

def fig0():
    figInd=0
    plt.figure(figInd)
    figInd+=1
    plt.plot(frames, rewards)
    plt.xlabel('Frames')
    plt.ylabel('Rewards')
    plt.ylim(-21, 21)
    
    plt.figure(figInd)
    figInd+=1
    plt.plot(frames, stds)
    plt.xlabel('Frames')
    plt.ylabel('std')
    
def fig1():
    figInd=0
    plt.figure(figInd)
    figInd+=1
    plt.plot(frames, rewards, label='fixed $\epsilon$')
    plt.plot(frames1, rewards1, label= 'decreased $\epsilon$')

    plt.xlabel('Frames')
    plt.ylabel('Rewards')
    plt.ylim(-21, 21)
    plt.legend()
    
    plt.figure(figInd)
    figInd+=1
    plt.plot(frames, stds)
    plt.xlabel('Frames')
    plt.ylabel('std')
   
fig1()   
plt.show()
