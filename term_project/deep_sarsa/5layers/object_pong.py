#######################################################################
# 06-27-18
# hhuang@math.fsu.edu
# object+rl for playing pong based on gym
# adam is not good for this one
# 07-13-18
# .gently decrease epsilon from 1 to 0.02, and keep 0.02 after 100k frames
# .adjust min/max, paddle opponent 4, paddle agent 12, ball 6
# .lam increase from 0.9 to 0.95
# . game specific, adjust cases when ball suddenly disappear
# 07-25-18
# . change the state to [dxb,dyb,xb,yb,ya]
# 07-26-18
# . change to frameskip4
# . by observing animations, paddle of agent velocity will have impact
#   on the effect of action, namely it has momentum effect, hence add
#   one state dimension of dy_agent
# 07-29-18
# . since test indicates ball should have high order as large as 10,
#   keep them as large as 10, while lower paddle order to 0~5, since 
#   test indicates higher orders are not important
# . further limit paddle order to accelerate
# 08-11-18 
# . by observing acceleration, find that it's almost always 0 for ball
#   only changes frequently for agent paddle, so only add states for
#   agent paddle acceleration
# 08-30-18
# . add 3rd action no-op
# . 3 frames for paddle of agent
# . split ball left, right action
# 12-08-18
# . NN for function approximation
#######################################################################

from __future__ import print_function
import gym
from gym import wrappers, logger
import numpy as np
import matplotlib.pyplot as plt

#env._max_episode_steps = 100000

# all possible actions
ACTION_UP = 2
ACTION_DOWN = 3
NO_OP=0
# order is important
ACTIONS = [NO_OP, ACTION_UP, ACTION_DOWN]

# bound for position and velocity
# bounds work as a normalization of the frequencies
# by observing 1000 frames, ball x or y difference in contiguous
#     frames is within [-4, 4], paddle y difference is also within it. 
#0-1 for opponent, 0 for position difference in height
#                1 for position in height at frame I
#2-5 for ball, 2 for position difference in width
#              3 for position difference in height
#              4 for position in width
#              5 for position in height 
#6-7 for agent, 6 for position difference in height
#               7 for position in height at frame I
# PADDLE_MIN = -3.5
# PADDLE_MAX = 79+3.5
# PADDLE_DIFF_MIN=-8.
# PADDLE_DIFF_MAX=8.
# BALL_X_MAX = 79.
# BALL_X_MIN = 0.
# BALL_Y_MAX = 79.5
# BALL_Y_MIN = -0.5
# BALL_X_DIFF_MAX = 4.
# BALL_X_DIFF_MIN = -4.
# BALL_Y_DIFF_MAX = 4.
# BALL_Y_DIFF_MIN = -4.
STATE_MIN=np.array([ -6.,  0.,  -0.5, -12, -12, -1.5])
STATE_MAX=np.array([ 6.,  79.,  79.5,  12,  12, 81.5])

#domain constants
COLOR_OPPONENT=213
COLOR_AGENT=92
COLOR_BALL=236
HEIGHT_PADDLE=8
HEIGHT_BALL=2
WIDTH_BALL=1
# width of paddle is 2, and this is the x-position for the left point
X_POSITION_OPPONENT=8
X_POSITION_AGENT=70

DISCOUNT = 0.95

# decrease it till 0.02
#EPSILON = 1.

# maximum steps per episode
STEP_LIMIT = 100000

def getObjectPosition(I, height, color, xPosition=None):
    position=0.
    found=False
    if xPosition is not None:
        #for paddle
        # case: paddle might be not fully exposed
        if(I[0,xPosition]!=0):
            found=True
            for j in range(1,1+height):
                if(I[j,xPosition]!=color):
                    position=j-1-(height/2.0-0.5)
                    break
        elif(I[79,xPosition]!=0):
            found=True
            for j in range(78,78-height,-1):
                if(I[j,xPosition]!=color):
                    position=j+1+height/2.0-0.5
                    break
        else:
            for j in range(1,80):
                if(I[j,xPosition]==color):
                    found=True
                    position=j+height/2.0-0.5
                    break
        return found, position
    else:
        #for ball
        position=np.zeros(2, dtype=int)
        for i in range(80):
            if found:
                break
            else:
                for j in range(80):
                    if(I[i,j]==color):
                        found=True
                        position[0]=i
                        position[1]=j
                        break
        if not found:
            return found, np.zeros(2)
        if position[0]==0:
            if I[position[0]+1,position[1]]!=color:
                return found, np.array([position[1], position[0]-0.5 ])
            return found, np.array([position[1], position[0]+0.5 ])
        elif position[0]==79:
            if I[position[0]-1,position[1]]!=color:
                return found, np.array([ position[1], position[0]+0.5])
            return found, np.array([ position[1], position[0]-0.5])
        return found, np.array([position[1], position[0]+0.5])

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    #hhuang: 210 height, 160 width, 
    I = I[35:195] # crop
    #hhuang: I=I[35:195,:,:], and here 195 is not included
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    if 1==0:
        for i in range(80):
            for j in range(80):
                #if(I[i,j]!=0):
                #if(I[i,j]==213 or I[i,j]==92):
                if(I[i,j]==213):
                    print('I[',i,',',j,']=',I[i,j])
                    break
    position=np.zeros(3)
    #opponentFound, position[0]=getObjectPosition(I, HEIGHT_PADDLE, COLOR_OPPONENT, X_POSITION_OPPONENT)
    ballFound, position[0:2]=getObjectPosition(I, HEIGHT_BALL, COLOR_BALL)
    agentFound, position[2]=getObjectPosition(I, HEIGHT_PADDLE, COLOR_AGENT, X_POSITION_AGENT)
    found=ballFound and agentFound
    #print('position: ', position)
    return found, position

def extract_state(position, positionOld=None, positionOlder=None):
    #0-2 for ball, 0 for position difference in height
    #              1 for position in width
    #              2 for position in height 
    #3-5 for agent, 3 for position difference in height at frame t-2
    #               4 for position difference in height at frame t-1
    #               5 for position in height at frame t
    goRight=0
    state=np.zeros(6)
    state[1]=position[0]
    state[2]=position[1]
    state[5]=position[2]
    #get opponent paddle info
    if positionOld is not None:
        if position[0]-positionOld[0]>=0:
            goRight=1
        #elif: position[0]-positionOld[0]<0:
        #    goRight=-1
        state[0]=position[1]-positionOld[1]
        state[4]=position[2]-positionOld[2]
        if positionOlder is not None:
            state[3]=positionOld[2]-positionOlder[2]

    #normalization to [0,1]
    state=(state-STATE_MIN)/(STATE_MAX-STATE_MIN)
    #if positionOld is not None:
    #    print('position old:', positionOld)
    #print('state', state)
    return goRight, state

# wrapper class for Sarsa(lambda)
class Sarsa_lambda:
    def __init__(self, stepSize, lam, dimension=6,
                 epsilon=1., H=128):
        self.dimension=dimension
        self.lam = lam
        self.stepSize = stepSize
        self.H=H
        # 2 for left/right, 3 for 3 actions
        self.weights={}
        self.weights['w1'] =(np.random.randn(2, 3, self.H, dimension)/
                             np.sqrt(dimension))
        self.weights['b1'] = np.ones((2, 3, self.H))

        self.weights['w2'] =(np.random.randn(2, 3, self.H, self.H)/
                             np.sqrt(self.H))
        self.weights['b2'] = np.ones((2, 3, self.H))

        self.weights['w3'] =(np.random.randn(2, 3, self.H, self.H)/
                             np.sqrt(self.H))
        self.weights['b3'] = np.ones((2, 3, self.H))

        self.weights['w4'] =(np.random.randn(2, 3, self.H, self.H)/
                             np.sqrt(self.H))
        self.weights['b4'] = np.ones((2, 3, self.H))

        self.weights['w5'] = np.random.randn(2, 3, 1, self.H)/np.sqrt(self.H)
        self.weights['b5'] = np.ones((2, 3, 1))

        self.traces={k:np.zeros_like(v) for k,v in
                     self.weights.items()}
        self.frame=0
        self.EPSILON=epsilon
        #self.loadW()

    def outputW(self, ep):
        np.savez_compressed('w_ep'+str(ep), a=self.weights['w1'], 
                                            b=self.weights['b1'], 
                                            c=self.weights['w2'], 
                                            d=self.weights['b2'],
                                            e=self.weights['w3'], 
                                            f=self.weights['b3'],
                                            g=self.weights['w4'], 
                                            h=self.weights['b4'],
                                            i=self.weights['w5'], 
                                            j=self.weights['b5'])

    def loadW(self, ep=1000):
        loaded=np.load('w_ep'+str(ep)+'.npz')
        self.weights['w1']=loaded['a']
        self.weights['b1']=loaded['b']
        self.weights['w2']=loaded['c']
        self.weights['b2']=loaded['d']
        self.weights['w3']=loaded['e']
        self.weights['b3']=loaded['f']
        self.weights['w4']=loaded['g']
        self.weights['b4']=loaded['h']
        self.weights['w5']=loaded['i']
        self.weights['b5']=loaded['j']
        print('finish loading weights at ep', ep)

    def reset(self):
        self.traces={k:np.zeros_like(v) for k,v in
                     self.weights.items()}

    def forward(self, x, action, goRight):
        act=action
        if act>0:
            act-=1
        h1=(self.weights['w1'][goRight, act].dot(x)+
            self.weights['b1'][goRight, act])
        h1[h1<0]=0. #reLU

        h2=(self.weights['w2'][goRight, act].dot(h1)+
            self.weights['b2'][goRight, act])
        h2[h2<0]=0. #reLU

        h3=(self.weights['w3'][goRight, act].dot(h2)+
            self.weights['b3'][goRight, act])
        h3[h3<0]=0. #reLU

        h4=(self.weights['w4'][goRight, act].dot(h3)+
            self.weights['b4'][goRight, act])
        h4[h4<0]=0. #reLU

        h5=(self.weights['w5'][goRight, act].dot(h4)+
            self.weights['b5'][goRight, act])
        q=h5[0]
        return h1, h2, h3, h4, q 

    def traceUpdate(self, x, h1, h2, h3, h4, action, goRight):
        #trace should get decayed in all action, not just the current action
        act=action
        if act>0:
            act-=1
        for k,v in self.traces.items():
            v*=self.lam*DISCOUNT

        self.traces['w5'][goRight, act, 0]+=h4
        self.traces['b5'][goRight, act, 0]+=1.

        dqdh4=self.weights['w5'][goRight, act, 0]
        dqdh4[h4<=0]=0.
        self.traces['w4'][goRight, act]+=np.outer(dqdh4, h3)
        self.traces['b4'][goRight, act]+=dqdh4

        dqdh4=np.reshape(dqdh4, (1, dqdh4.shape[0]))
        dqdh3=dqdh4.dot(self.weights['w4'][goRight, act])
        dqdh3=np.ravel(dqdh3)
        dqdh3[h3<=0]=0.
        self.traces['w3'][goRight, act]+=np.outer(dqdh3, h2)
        self.traces['b3'][goRight, act]+=dqdh3

        dqdh3=np.reshape(dqdh3, (1, dqdh3.shape[0]))
        dqdh2=dqdh3.dot(self.weights['w3'][goRight, act])
        dqdh2=np.ravel(dqdh2)
        dqdh2[h2<=0]=0.
        self.traces['w2'][goRight, act]+=np.outer(dqdh2, h1)
        self.traces['b2'][goRight, act]+=dqdh2

        dqdh2=np.reshape(dqdh2, (1, dqdh2.shape[0]))
        dqdh1=dqdh2.dot(self.weights['w2'][goRight, act])
        dqdh1=np.ravel(dqdh1)
        dqdh1[h1<=0]=0.
        self.traces['w1'][goRight, act]+=np.outer(dqdh1, x)
        self.traces['b1'][goRight, act]+=dqdh1

    # learn with given state, action and target
    def learn(self, goRight, x, action, reward, newGoRight, newX, newAction, ep,
             justScored=False):
        #for i in range(self.pieces):
            #print('phi[',i,']=',phi[i], 'newPhi[',i,']=',newPhi[i])
        h1, h2, h3, h4, estimation = self.forward(x, action, goRight)
        h1_, h2_, h3_, h4_, newEst = self.forward(newX, newAction, newGoRight)
        if justScored:
            newEst=0.
        delta = reward+ DISCOUNT*newEst - estimation
        self.traceUpdate(x, h1, h2, h3, h4, action, goRight)
        for k, v in self.traces.items():
            update=self.stepSize * delta * v
            if ep % 100==0 and self.frame % 100==0:
                updateScale=np.linalg.norm(update)
                weightsScale=np.linalg.norm(self.weights[k])
                if weightsScale==0.:
                    print(k, 'parame scale: ', weightsScale)
                else:
                    print(k, 'update scale/ parame scale: ', updateScale/weightsScale)
                print('qOld ', estimation, 'q ', newEst, 'delta ', delta)
            self.weights[k] += update

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def getAction(x, goRight, valueFunction):
    if np.random.binomial(1, valueFunction.EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        h1_, h2_, h3_, h4_, value = valueFunction.forward(x, action, goRight)
        values.append(value)
    #if valueFunction.frame % 100==0:
    #    print('action values: ', values)
    #hh: return argmax(values) - 1
    action= np.random.choice([action_ for action_, value_ in
                              enumerate(values) if value_ == np.max(values)])
    #print('values: ',values)
    #print('state: ', observation, 'action: ', action)
    if action==0: return NO_OP
    else: return action+1

def play(evaluator, env, ep):
    observation=env.reset()
    #print(observation)
    done=False
    state=None
    phi=None
    rewardSum=0.
    found=False
    while not done:
        # game specific, neglect beginning and endding frames
        justScored=False
        found=False
        position=None
        oldPosition=None
        newGoRight=0
        goRight=0
        while not found:
            action = NO_OP
            #env.render()
            #env.env.ale.saveScreenPNG(b'pong_'+str(1000+evaluator.frame)+'.png')
            newObservation, reward, done, info=env.step(action)
            found, oldPosition=prepro(newObservation)
        newObservation, reward, done, info=env.step(NO_OP)
        found, position=prepro(newObservation)
        newObservation, reward, done, info=env.step(NO_OP)
        found, newPosition=prepro(newObservation)

        # now found, game begins
        goRight, state=extract_state(newPosition, position, oldPosition)
        oldposition=np.copy(position)
        position=np.copy(newPosition)
        action = getAction(state, goRight, evaluator)
        reward=0.
        while reward==0.:
            #env.env.ale.saveScreenPNG(b'pong_'+str(1000+evaluator.frame)+'.png')
            #print('frame', evaluator.frame)
            evaluator.frame+=1
            #env.render()
            newObservation, reward, done, info=env.step(action)
            if reward!=0.:
                justScored=True
            #when reward=-1 or 1, ball will disappear
            found, newPosition=prepro(newObservation)
            #game specific, when ball hit the upper wall, it will
            #disappear sometimes and then re-appear again
            if not found:
                #keep using the old ball width position and motion info
                newPosition=np.copy(position)
                if goRight==1:
                    newPosition[0]+=2.
                else:
                    newPosition[0]-=2.
            newGoRight, newState=extract_state(newPosition, position, oldPosition)
            #print('new state:', state)
            # this phi is corresponding to state, no action 
            newAction = getAction(state, newGoRight, evaluator)
            #newAction = getAction(newObservation, evaluator)
            evaluator.learn(goRight, state, action, reward, newGoRight, 
                            newState, newAction, ep, justScored)
            oldPosition=np.copy(position)
            position=np.copy(newPosition)
            goRight=newGoRight
            state = np.copy(newState)
            action = newAction
        rewardSum+=reward
        print('reward: ', reward, ' scores: ', rewardSum)
        #once non-zero score happened, the ball will disappear, no longer found
        evaluator.reset()
    evaluator.reset()
    return rewardSum

def figure_order_effect():
    runs = 1
    episodes = 10001
    #alphas = np.arange(1, 2) / 20000.0
    #alphas = [0.003, 0.01]
    #alphas = [0.001]
    alphas = [0.0005]
    #lams = [0.99, 0.98, 0.9, 0.8, 0.7]
    lams = [0.95]
    rewards = np.zeros((len(lams), len(alphas), runs, episodes))

    #logger.set_level(logger.INFO)
    #video_path='/home/hh/Dropbox/Hua/cart_pole/movie/'
    #env=gym.make('Pong-v0')
    env=gym.make('PongDeterministic-v0')
    #env=wrappers.Monitor(env, directory=video_path, force=True)
    f=open('output.dat', 'w')
    f.close()
    for lamInd, lam in enumerate(lams):
        for alphaInd, alpha in enumerate(alphas):
            for run in range(runs):
                evaluator = Sarsa_lambda(alpha, lam)
                for ep in range(episodes):
                    reward = play(evaluator, env, ep)
                    f=open('output.dat', 'a')
                    out=str(ep)+' '+str(reward)+' '+str(evaluator.frame)+'\n'
                    f.write(out)
                    f.close()
                    #step = play_forgetful_ls_sarsa_lambda(evaluator, ep+1)
                    if ep%1==0: print('episode %d, scores %d, frame %d' %
                                      (ep, reward, evaluator.frame))
                    if ep%1000==0: evaluator.outputW(ep)
                    #if ep>0 and ep%1000==0: evaluator.stepSizes*=0.5
                    if ep<4500:
                        if evaluator.frame>100000:
                            evaluator.EPSILON=0.05
                        else:
                            evaluator.EPSILON=2./(1.+np.exp(evaluator.frame/30000.))
                    else:
                        #if ep>=4500 and ep%500==0:
                        if ep>=9000 and ep%500==0:
                            evaluator.EPSILON*=0.5
                    #if(ep+1 in episodesToPlot): plot_costToGo(evaluator, ep+1)

    #env.close()

if __name__ == '__main__':
    #env.env.ale.saveScreenPNG(b'pong.png')
    #env.env.ale.saveScreenPNG(b'pong_'+str(1000+index)+'.png')
    figure_order_effect()
    #figure12_10()
    #figure12_11()
    #plt.show()



