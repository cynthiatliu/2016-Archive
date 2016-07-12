#Tic tac toe
#How to implement RL

import numpy as np
from sknn.mlp import Regressor, Convolution, Layer
# import q_network
# import pyrl.agents.policy_gradient_cl as pg
import math
import random
import sys
# import tensorflow as tf

def getNextState(state, action, par):
    move = actions[action]
    x = move[0]; y = move[1]
    
    if par%2 == 1: state[x,y] = 1
    else: state[x,y] = -1

    return newState

def getReward(state):
    for i in range(3): #extract rows
        if [1,1,1] == state[i]: #ith win state in this case is just winning on the ith row
            return 1

    for i in range(3): #extract columns
        state_c = [state[0][i], state[1][i], state[2][i]]
        if [1,1,1] == state_c: return 1

    #top left to bottom right
    state_d1 = [state[0][0], state[1][1], state[2][2]]
    if [1,1,1] == state_d1: return 1

    #top right to bottom left
    state_d2 =[state[0][2], state[1][1], state[2][0]]
    if [1,1,1] == state_d2: return 1

    return 0

def getLoss(state):
    for i in range(3): #extract rows
        if [-1,-1,-1] == state[i]: #ith win state in this case is just winning on the ith row
            return -1

    for i in range(3): #extract columns
        state_c = [state[0][i], state[1][i], state[2][i]]
        if [-1,-1,-1] == state_c: return -1

    #top left to bottom right
    state_d1 = [state[0][0], state[1][1], state[2][2]]
    if [-1,-1,-1] == state_d1: return -1

    #top right to bottom left
    state_d2 =[state[0][2], state[1][1], state[2][0]]
    if [-1,-1,-1] == state_d2: return -1

    return 0

def main():

    #Initialize the board - the state
    state = np.ndarray(shape=(3,3), dtype=int)
    actions = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]] #also spaces
    moveNum = 1

    #Variables - might not be necessary
    #k = 1
    #alpha = 1/k
    gamma = .3
    #eps = .1
    tau = .02

    critic = Regressor(layers=[Layer("Rectifier", name="layer1", units=11, pieces=2), #9 squares, 1 action, 1 bias
                               Layer("Rectifier", name="layer2", units=11, pieces=2),
                               Layer("Rectifier", name="layer3", units=11, pieces=2),
                               Layer("Softmax")], random_state=1, learning_rate=0.02)
    
    #Randomly initialize the critic
    statesAndActs = []; rewards = []
    for i in range(500):
        sample_st = np.ndarray(shape=(3,3), dtype=int)
        for j in range(9):
            sample_st[math.floor(j/3),j%3] = int(random.randint(-1,1))
            
        if i == 0:
            print sample_st

        #random action, random reward
        act = random.randint(0,8)
        rew = random.randint(-1,1)
        rewards.append(rew)
        
        stateAndAct = []
        for k in range(9):
            stateAndAct.append(sample_st[int(k/3),k%3])
            
        stateAndAct.append(act)
        if i == 0:
            print stateAndAct
        statesAndActs.append(stateAndAct)
    
    print "hi"
    sA = np.array(statesAndActs); r = np.array(rewards)
    critic.fit(sA,r)
    print "aloha"
        
    target_Q = critic

    #Training
    for i in range(1000):
        reward = 0; end = False
        R = [] #Replay buffer

        while (end != True):

            #We play as both
            x = -1; y = -1
            success = False
            while success == False:
                try:
                    x, y = int(input("Enter the row and column indices of the location at which you intend to draw an 'X.' (Format: x, y):    "))
                    action = actions.index([x,y])
                    if action in actions: 
                        success = True
                        
                except:
                    print ("I'm sorry, but x and y should be numerical.")
                    
            newstate = getNextState(state, action, moveNum) #Execute action
            moveNum = moveNum + 1

            #Observe reward
            reward = getReward(newstate)
            R.append(state, action, reward, newstate)
            if reward != 0: #Game is done
                end = True
                break;

            N = math.floor(math.log(len(R)))
            R2 = R; minibatch = []
            for i in range(N):
                j = random.randint(0,len(R2)-1)
                minibatch.append(R2[j])
                R2.remove(R2[j])

            ys = []; batchStates = []
            for i in range(N):
                s_1 = minibatch[i][3]; r = minibatch[i][2]
                ys.append(r + gamma*target_Q.predict(s_1))

                #Make new input for retraining - includes state and action
                batchStates.append([minibatch[i][0],minibatch[i][0]])

            #minimize the loss L = (1/N)*sum(ys[i] - critic.predict(state))^2 - a linear regression
            if len(batchStates) != 0:
                critic.fit(batchStates,ys)

            #Update the target network manually
            Q_para = np.array(critic.get_parameters())
            if i == 0:
                target_Q.set_parameters(Q_para)
            else:
                Qp_para = np.array(target_Q.get_parameters())
                new_para = tau*Q_para + (1-tau)*Qp_para
                target_Q.set_parameters(new_para)
            

            #Set state to the new state.
            state = newstate    

            #"O"
            x = -1; y = -1
            while success == True:
                try:
                    x, y = int(input("Enter the row and column indices of the location at which you intend to draw an 'O.' (Format: x, y):    "))
                    action = [x,y]
                    if action in actions: 
                        success = False
                except:
                    print ("I'm sorry, but x and y should be numerical.")

            newstate2 = getNextState(state, action, par)
            reward = getLoss(state)
            R.append(state, action, reward, newstate2)
            if reward != 0: #Game is done
                end = True
                break;

    #end While
    
    #Testing the critic
    for i in range(10):
        print "hi"
    print "Meow"

main()