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
    def __init__(self, stepSize, lam, dimension=6, order=10,
            beta2=0.999, epsilon=1.):
        self.pieces = np.power(order+1,dimension)
        self.order=order
        self.dimension=dimension
        self.lam = lam
        self.stepSize = stepSize
        self.c=np.zeros((self.pieces, self.dimension))
        self.cal_coeff_c()

        self.weights = np.zeros((2, 3*self.pieces))
        self.stepSizes = np.ones(3*self.pieces)*stepSize
        self.cal_stepSize()
        self.trace = np.copy(self.weights)
        self.frame=0
        self.EPSILON=epsilon

    def outputW(self, ep):
        np.savez_compressed('w_ep'+str(ep), a=self.weights)

    def cal_coeff_c(self):
        #c=np.zeros((pieces,dimension))
        coeff=[]
        for i in range(self.pieces):
            index=i
            c=np.zeros(self.dimension)
            #for j in range(self.dimension):
            for j in range(6):
                c[j]=index%(self.order+1)
                index //=(self.order+1)
            #if c[3]<=5 and c[4]<=5 and c[5]<=5 and c[4]+c[5]<=5:
            if c[3]<=5 and c[4]<=5 and c[5]<=5:
                coeff.append(c)
        piecesUpdated=len(coeff)
        print('total pieces: ', piecesUpdated)
        self.c=np.vstack(coeff)
        self.pieces=piecesUpdated
            #print('c[',i,']=',c[i])
        #return c

    def cal_features_fourier(self, state):
        #phi=np.zeros(self.pieces)
        #for i in range(self.pieces):
        #    phi[i]=np.cos(np.pi*np.sum(state*self.c[i]))
        phi=np.cos(np.pi*np.sum(self.c*state, axis=1))
        return phi

    def reset(self):
        self.trace = np.zeros_like(self.trace)

    def cal_stepSize(self):
        for action in ACTIONS:
            index0=0
            if action>0: index0=int((action-1)*self.pieces)    
            for i in range(self.pieces):
                cNorm=np.sqrt(np.mean(self.c[i]*self.c[i]))
                if(cNorm!=0): self.stepSizes[index0+i]/=cNorm

    # estimate the value of given state and action
    def value(self, phi, action, goRight):
        if action==NO_OP:
            #for i in range(self.pieces):
            #    print('w[',i,']=',self.weights[i], 'phi', phi[i])
            return np.sum(self.weights[goRight, :self.pieces]*phi)
        elif action==ACTION_UP:
            #for i in range(self.pieces):
            #    print('w[',i+self.pieces,']=',self.weights[i+self.pieces], 'phi', phi[i])
            return np.sum(self.weights[goRight, self.pieces:2*self.pieces]*phi)
        else:
            return np.sum(self.weights[goRight, 2*self.pieces:]*phi)

    def traceUpdate(self, phi, action, goRight):
        self.trace *= self.lam * DISCOUNT
        if action==NO_OP:
            self.trace[goRight, :self.pieces] += phi
        elif action==ACTION_UP:
            self.trace[goRight, self.pieces:2*self.pieces] += phi
        else:
            self.trace[goRight, 2*self.pieces:] += phi

    # learn with given state, action and target
    def learn(self, goRight,  phi, action, reward, newGoRight, newPhi, newAction, ep,
             justScored=False):
        #for i in range(self.pieces):
            #print('phi[',i,']=',phi[i], 'newPhi[',i,']=',newPhi[i])
        estimation = self.value(phi, action, goRight)
        newEst = self.value(newPhi, newAction, newGoRight)
        if justScored:
            newEst=0.
        delta = reward+ DISCOUNT*newEst - estimation
        self.traceUpdate(phi, action, goRight)
        #for i in range(2*self.pieces):
            #print('z[',i,']=',self.trace[i])
        update=self.stepSizes * delta * self.trace
        if ep % 100==0 and self.frame % 100==0:
            updateScale=np.linalg.norm(update)
            weightsScale=np.linalg.norm(self.weights)
            if weightsScale==0.:
                print('parame scale: ', weightsScale)
            else:
                print('update scale/ parame scale: ', updateScale/weightsScale)
            print('qOld ', estimation, 'q ', newEst, 'delta ', delta)
        self.weights += update

# get action at @position and @velocity based on epsilon greedy policy and @valueFunction
def getAction(phi, goRight, valueFunction):
    if valueFunction.frame>100000:
        valueFunction.EPSILON=0.05
    else:
        valueFunction.EPSILON=2./(1.+np.exp(valueFunction.frame/30000.))

    if np.random.binomial(1, valueFunction.EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(valueFunction.value(phi, action, goRight))
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
            #print('not found')
            #action = np.random.choice(ACTIONS)
            action = NO_OP
            #env.render()
            #env.env.ale.saveScreenPNG(b'pong_'+str(1000+evaluator.frame)+'.png')
            #print('frame', evaluator.frame)
            #evaluator.frame+=1
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
        # this phi is corresponding to state, no action 
        phi=evaluator.cal_features_fourier(state)
        action = getAction(phi, goRight, evaluator)
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
            newGoRight, state=extract_state(newPosition, position, oldPosition)
            #if evaluator.frame%100==0:
            #    print( 'old position:', position)
            #    print('new position:', newPosition)
            #print('new state:', state)
            # this phi is corresponding to state, no action 
            newPhi=evaluator.cal_features_fourier(state)
            #print(newObservation[1], newObservation[3])
            #print(newObservation)
            newAction = getAction(newPhi, newGoRight, evaluator)
            #newAction = getAction(newObservation, evaluator)
            evaluator.learn(goRight, phi, action, reward, newGoRight, newPhi, newAction, ep, justScored)
            oldPosition=np.copy(position)
            position=np.copy(newPosition)
            goRight=newGoRight
            phi = np.copy(newPhi)
            action = newAction
        rewardSum+=reward
        print('reward: ', reward, ' scores: ', rewardSum)
        #once non-zero score happened, the ball will disappear, no longer found
        evaluator.reset()
    evaluator.reset()
    return rewardSum

def figure_order_effect():
    runs = 1
    episodes = 10000
    #alphas = np.arange(1, 2) / 20000.0
    #alphas = [0.003, 0.01]
    alphas = [0.00001]
    #lams = [0.99, 0.98, 0.9, 0.8, 0.7]
    orders=[10]
    lams = [0.95]
    rewards = np.zeros((len(orders), len(lams), len(alphas), runs, episodes))

    #logger.set_level(logger.INFO)
    #video_path='/home/hh/Dropbox/Hua/cart_pole/movie/'
    #env=gym.make('Pong-v0')
    env=gym.make('PongDeterministic-v0')
    #env=wrappers.Monitor(env, directory=video_path, force=True)
    for orderInd, order in enumerate(orders):
        for lamInd, lam in enumerate(lams):
            for alphaInd, alpha in enumerate(alphas):
                for run in range(runs):
                    evaluator = Sarsa_lambda(alpha, lam, order=order)
                    for ep in range(episodes):
                        reward = play(evaluator, env, ep)
                        #step = play_forgetful_ls_sarsa_lambda(evaluator, ep+1)
                        rewards[orderInd, lamInd, alphaInd, run, ep] = reward
                        if ep%1==0: print('episode %d, scores %d,frame %d' %
                                          (ep, reward, evaluator.frame))
                        if ep%100==0: evaluator.outputW(ep)
                        if ep>0 and ep%1000==0: evaluator.stepSizes*=0.5
                        #if(ep+1 in episodesToPlot): plot_costToGo(evaluator, ep+1)

    #env.close()

    np.savez_compressed('sarsa_lambda_order10_lam0.95_alpha0.00001_run1', a=rewards)

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



