import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('output.dat')
m=data.shape[0]
frames=np.zeros(m//100)
rewards=np.zeros(m//100)
stds=np.zeros(m//100)
for i in range(50, m-50, 100):
    rewards[i//100]=np.mean(data[i-50:i+50,1])
    stds[i//100]=np.std(data[i-50:i+50,1])
    frames[i//100]=data[i, 2]
frames=frames[1:]
rewards=rewards[1:]
stds=stds[1:]

figInd=0
plt.figure(figInd)
figInd+=1
plt.plot(frames, rewards)
plt.xlabel('Frames')
plt.ylabel('Rewards')
plt.ylim(-21,21)

plt.figure(figInd)
figInd+=1
plt.plot(frames, stds)
plt.xlabel('Frames')
plt.ylabel('std')

plt.show()
