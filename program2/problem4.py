import numpy as np

#bias=np.genfromtxt('hw2_softmax_weights.m', skip_footer=100,  delimiter=' ')
#weight=np.genfromtxt('hw2_softmax_weights.m', skip_header=1,  delimiter=' ')
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
print('w.T[0]', temp[0])
print('w.T[18]', temp[18])
print('norms of w.T:')
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


