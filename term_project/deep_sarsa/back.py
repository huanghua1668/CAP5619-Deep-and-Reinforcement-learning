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
# 07-18-18 deep net work
#######################################################################

from __future__ import print_function
import gym
from gym import wrappers, logger
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import MaxNLocator, LinearLocator, FormatStrFormatter

#env._max_episode_steps = 100000

# all possible actions
ACTION_UP = 2
ACTION_DOWN = 3
# order is important
ACTIONS = [ACTION_UP, ACTION_DOWN]

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
STATE_MIN=np.array([-4., -3.5, -6., -6., 
                       0.,  -0.5, -12., -3.5])
STATE_MAX=np.array([ 4., 82.5, 6., 6., 
                       79.,  79.5, 12., 82.5])

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
                return found, np.array([position[0]-0.5, position[1]])
            return found, np.array([position[0]+0.5, position[1]])
        elif position[0]==79:
            if I[position[0]-1,position[1]]!=color:
                return found, np.array([position[0]+0.5, position[1]])
            return found, np.array([position[0]-0.5, position[1]])
        return found, np.array([position[0]+0.5, position[1]])

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
    position=np.zeros(4)
    opponentFound, position[0]=getObjectPosition(I, HEIGHT_PADDLE, COLOR_OPPONENT, X_POSITION_OPPONENT)
    ballFound, position[1:3]=getObjectPosition(I, HEIGHT_BALL, COLOR_BALL)
    agentFound, position[3]=getObjectPosition(I, HEIGHT_PADDLE, COLOR_AGENT, X_POSITION_AGENT)
    found=opponentFound and ballFound and agentFound
    #print('position: ', position)
    return found, position

def extract_state(position, positionOld=None):
    #0-1 for opponent, 0 for position difference in height
    #                1 for position in height at frame I
    #2-5 for ball, 2 for position difference in width
    #              3 for position difference in height
    #              4 for position in width
    #              5 for position in height 
    #6-7 for agent, 6 for position difference in height
    #               7 for position in height at frame I
    state=np.zeros(8)
    state[1]=position[0]
    state[4]=position[2]
    state[5]=position[1]
    state[7]=position[3]
    #get opponent paddle info
    if positionOld is not None:
        state[0]=position[0]-positionOld[0]
        state[2]=position[2]-positionOld[2]
        state[3]=position[1]-positionOld[1]
        state[6]=position[3]-positionOld[3]

    #normalization to [0,1]
    state=(state-STATE_MIN)/(STATE_MAX-STATE_MIN)
    #if positionOld is not None:
    #    print('position old:', positionOld)
    #print('state', state)
    return state

# wrapper class for Sarsa(lambda)
class Sarsa_lambda_deep:
    def __init__(self, stepSize, lam, dimension=8, epsilon=1.):
        self.dimension=dimension
        self.lam = lam
        self.stepSize = stepSize
        self.H=200
        self.weights={}
        self.weights['w1'] = np.random.randn(self.H, dimension)/np.sqrt(dimension)
        self.weights['b1'] = np.random.randn(self.H)/np.sqrt(self.H)
        self.weights['w2'] = np.random.randn(2, self.H)/np.sqrt(self.H)
        self.weights['b2'] = np.random.randn(2)/np.sqrt(2.)
        self.trace={k:np.zeros_like(v) for k,v in
                     self.weights.iteritems()}
        self.frame=0
        self.EPSILON=epsilon

    def outputW(self, ep):
        np.savez_compressed('w_ep'+str(ep), a=self.weights['w1'], 
                                            b=self.weights['b1'], 
                                            c=self.weights['w2'], 
                                            d=self.weights['b2'] )

    def reset(self):
        self.trace={k:np.zeros_like(v) for k,v in
                     self.weights.iteritems()}

    def forward(self, x, actionUp=None, actionDown=None):
        h=np.zeros(self.H)
        h=self.weights['w1'].dot(x)+self.weights['b1']
        h[h<0]=0. #reLU
        qUp=0.
        qDown=0.
        if actionUp is not None:
            index=0
            qUp=self.weights['w2'][index].dot(h)+self.weights['b2'][index]
        if actionDown is not None:
            index=1
            qDown=self.weights['w2'][index].dot(h)+self.weights['b2'][index]
        return h, qUp, qDown

    # estimate the value of given state and action
    def value(self, state, action):
        index=action-2
        if action==ACTION_UP:
            h, value, value_=self.forward(state,actionUp=action)
        else:
            h, value_, value=self.forward(state,actionDown=action)
        return value, h

    def traceUpdate(self, x, action, h):
        index=action-2
        #self.trace['b2'][index]*=self.lam * DISCOUNT
        #trace should get decayed in all action, not just the current action
        self.trace['b2']*=self.lam * DISCOUNT
        self.trace['b2'][index]+=1.
        #self.trace['w2'][index]*=self.lam * DISCOUNT
        self.trace['w2']*=self.lam * DISCOUNT
        self.trace['w2'][index]+=h
        dh=self.weights['w2'][index]
        dh[h<=0.]=0. #backpro reLU
        self.trace['b1'] *=self.lam * DISCOUNT
        self.trace['b1'] +=dh
        self.trace['w1'] *=self.lam * DISCOUNT
        self.trace['w1'] +=np.outer(dh,x)

    # learn with given state, action and target
    def learn(self, x, action, h, delta, ep):
        #for i in range(self.pieces):
            #print('phi[',i,']=',phi[i], 'newPhi[',i,']=',newPhi[i])
        self.traceUpdate(x, action, h)
        #for i in range(2*self.pieces):
            #print('z[',i,']=',self.trace[i])
        for k, v in self.weights.iteritems():
            update=self.stepSize * delta * self.trace[k]
            self.weights[k] += update
            if ep % 100==0 and self.frame % 100==0:
                updateScale=np.linalg.norm(update)
                weightsScale=np.linalg.norm(self.weights[k])
                if weightsScale==0.:
                    print('parame scale for ', k, ': ', weightsScale)
                else:
                    print('update scale/ parame scale for ', k,': ', updateScale/weightsScale)

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def getAction(state, valueFunction):
    if valueFunction.frame>100000:
        valueFunction.EPSILON=0.05
    else:
        valueFunction.EPSILON=2./(1.+np.exp(valueFunction.frame/30000.))

    values = []
    h_, qUp, qDown= valueFunction.forward(state, ACTION_UP, ACTION_DOWN)
    values.append(qUp)
    values.append(qDown)

    if np.random.binomial(1, valueFunction.EPSILON) == 1:
        action=np.random.choice(ACTIONS)
        return action, values[action-2]
    #if valueFunction.frame % 100==0:
    #    print('action values: ', values)
    #hh: return argmax(values) - 1
    action= np.random.choice([action_ for action_, value_ in
                              enumerate(values) if value_ == np.max(values)])
    #print('values: ',values)
    #print('state: ', observation, 'action: ', action)
    return action+2, values[action]

def play(evaluator, env, ep):
    observation=env.reset()
    #print(observation)
    done=False
    state=None
    rewardSum=0.
    found=False
    while not done:
        # game specific, neglect beginning and endding frames
        justScored=False
        found=False
        position=None
        while not found:
            #print('not found')
            action = np.random.choice(ACTIONS)
            #env.render()
            #env.env.ale.saveScreenPNG(b'pong_'+str(1000+evaluator.frame)+'.png')
            #print('frame', evaluator.frame)
            #evaluator.frame+=1
            newObservation, reward, done, info=env.step(action)
            found, newPosition=prepro(newObservation)
        # now found, game begins
        state=extract_state(newPosition, position)
        position=newPosition
        # this phi is corresponding to state, no action 
        action, estimate_= getAction(state, evaluator)
        reward=0.
        while reward==0.:
            estimate, h=evaluator.value(state, action)
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
                newPosition[2]=position[2]
                if state[2]>0:
                    newPosition[2]+=1.
                elif state[2]<0:
                    newPosition[2]-=1.
            newState=extract_state(newPosition, position)
            #if evaluator.frame%100==0:
            #    print( 'old position:', position)
            #    print('new position:', newPosition)
            #print('new state:', state)
            #print(newObservation[1], newObservation[3])
            #print(newObservation)
            newAction, newEstimate= getAction(newState, evaluator)
            if justScored:
                newEstimate=0.
            #newAction = getAction(newObservation, evaluator)
            delta=DISCOUNT*newEstimate+reward-estimate
            if rewardSum== 0 and evaluator.frame%100==0: 
                print('qOld ', estimate, 'q ', newEstimate, 'delta ', delta)
            evaluator.learn(state, action, h, delta, ep)
            position=np.copy(newPosition)
            state = np.copy(newState)
            action = newAction
        rewardSum+=reward
        #once non-zero score happened, the ball will disappear, no longer found
        evaluator.reset()
    return rewardSum

def figure_order_effect():
    runs = 1
    episodes = 100000
    #alphas = np.arange(1, 2) / 20000.0
    #alphas = [0.003, 0.01]
    alphas = [0.0001]
    #lams = [0.99, 0.98, 0.9, 0.8, 0.7]
    orders=[3]
    lams = [0.95]
    rewards = np.zeros((len(orders), len(lams), len(alphas), runs, episodes))

    #logger.set_level(logger.INFO)
    #video_path='/home/hh/Dropbox/Hua/cart_pole/movie/'
    env=gym.make('Pong-v0')
    #env=wrappers.Monitor(env, directory=video_path, force=True)
    for orderInd, order in enumerate(orders):
        for lamInd, lam in enumerate(lams):
            for alphaInd, alpha in enumerate(alphas):
                for run in range(runs):
                    evaluator = Sarsa_lambda_deep(alpha, lam)
                    for ep in range(episodes):
                        reward = play(evaluator, env, ep)
                        #step = play_forgetful_ls_sarsa_lambda(evaluator, ep+1)
                        rewards[orderInd, lamInd, alphaInd, run, ep] = reward
                        if ep%1==0: print('episode %d, scores %d,frame %d' %
                                          (ep, reward, evaluator.frame))
                        if ep%400==0: evaluator.outputW(ep)
                        #if(ep+1 in episodesToPlot): plot_costToGo(evaluator, ep+1)

    #env.close()

    np.savez_compressed('sarsa_lambda_lam0.95_alpha0.0001_run1', a=rewards)

    # average over runs
    #steps = np.mean(steps, axis=3)

    if 1==0:
        global figureIndex
        figureIndex=0
        plt.figure(figureIndex)
        figureIndex += 1
        for orderInd, order in enumerate(orders):
            plt.plot(orders, steps[:, 0, 0], label='order = %s' % (str(order)))
        plt.xlabel('order')
        plt.ylabel('averaged steps per episode')
        plt.ylim([80, 300])
        plt.legend()

if __name__ == '__main__':
    #env.env.ale.saveScreenPNG(b'pong.png')
    #env.env.ale.saveScreenPNG(b'pong_'+str(1000+index)+'.png')
    figure_order_effect()
    #figure12_10()
    #figure12_11()
    #plt.show()




