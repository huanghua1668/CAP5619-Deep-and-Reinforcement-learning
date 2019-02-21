import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('output.dat')
m=data.shape[0]
frames=np.zeros(m//50)
rewards=np.zeros(m//50)
stds=np.zeros(m//50)
for i in range(50, m-50, 50):
    rewards[i//50]=np.mean(data[i-50:i+50,1])
    stds[i//50]=np.std(data[i-50:i+50,1])
    frames[i//50]=data[i, 2]
frames=frames[1:]
rewards=rewards[1:]
stds=stds[1:]

figInd=0
plt.figure(figInd)
figInd+=1
plt.plot(frames, rewards)
plt.xlabel('Frames')
plt.ylabel('Rewards')

plt.figure(figInd)
figInd+=1
plt.plot(frames, stds)
plt.xlabel('Frames')
plt.ylabel('std')

plt.show()
