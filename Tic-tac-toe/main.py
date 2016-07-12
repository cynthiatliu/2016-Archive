#Tic tac toe
#How to implement RL

import numpy as np
from sknn.mlp import Regressor, Convolution, Layer
# import q_network
import pyrl.agents.policy_gradient_cl as pg
import math
import random
import sys

def getNextState(state, action, actions):
    move = actions[action]
    x = move[0]; y = move[1]
    state[x,y] = 1
    
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
    win_states = makeWinStates()
    actions = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]] #also spaces
    
    #Variables - might not be necessary
    k = 1
    alpha = 1/k
    gamma = .3
    eps = .1

    #Initializing our 'overkill' neural networks
    actor = Regressor(layers, warning=None, weights=None, random_state=None, 
                     learning_rule='sgd', 
                     learning_rate=0.01, 
                     learning_momentum=0.9, 
                     regularize=None, weight_decay=None, 
                     dropout_rate=None, batch_size=1, 
                     n_iter=None, n_stable=10, 
                     f_stable=0.001, valid_set=None, 
                     valid_size=0.0, loss_type=None, 
                     callback=None, debug=False, 
                     verbose=None) #???
    
    #Training the actor with a random policy
    trainStates = []; acts = []
    for i in range(500):
	sample_st = np.ndarray(shape=(3,3), dtype=int)
	for j in range(9):
	    sample_st[math.floor(j/3),j%3] = random.randint(-1,1)
	    
	act = random.randint(0,8) #action represented by its index
	trainStates.append(sample_st)
	acts.append(act)
	
    actor.fit(trainStates, acts)
    
    target_mu = actor
    
    
    critic = Regressor(layers=[Layer("Rectifier", name="layer1", units=11, pieces=2), #9 squares, 1 action, 1 bias
                       Layer("Rectifier", name="layer2", units=11, pieces=2),
                       Layer("Rectifier", name="layer3", units=11, pieces=2),
                       Layer("Softmax")], learning_rate=0.02)
    
    #Randomly initialize the critic
    statesAndActs = []; rewards = []
    for i in range(500):
	sample_st = np.ndarray(shape=(3,3), dtype=int)
	for j in range(9):
	    sample_st[math.floor(j/3),j%3] = random.randint(-1,1)

	#random action, random reward
	act = random.randint(0,8)
	rew = random.randint(-1,1)

	statesAndActs.append([sample_st,act])
	rewards.append(rew)

    critic.fit(statesAndActs,rewards)

    target_Q = critic

    
    for i in range(10):
	reward = 0; end = False; R = []
	
	while (end != True):

	    action = actor.predict(state)
	    newstate = getNextState(state, action) #Execute action
	    
	    #Observe reward
	    reward = getReward(state)
	    if reward != 0: #Game is done
		end = True		    
	    
	    #Replay buffer review
	    R.append(state, action, reward, newstate)
	    
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
	    
	    #update the actor policy somehow -- this is the hard part; test the critic alone first
	    
	    #Update the target critic
	    Q_para = np.array(critic.get_parameters())
	    if i == 0:
		target_Q.set_parameters(Q_para)
	    else:
		Qp_para = np.array(target_Q.get_parameters())
		new_para = tau*Q_para + (1-tau)*Qp_para
		target_Q.set_parameters(new_para)
		
	    #Update the target actor
	    
	    
	    #How do I write this
	    
	    #Set state to the new state.
	    state = newstate
	    
	    reward = getReward(state)
	    if reward != 0: #Game is done
		end = True	    
	    
	    #We play as "O"
	    x = -1; y = -1
	    while not (x >= 0 and x <= 2 and y >= 0 and y <= 2):
		try:
		    x, y = int(input("Enter the row and column indices of the location at which you intend to draw an 'O.' (Format: x, y):    "))
		    while x <= 0 or x >= 2 or y <= 0 or y >= 2:
			x, y = int(input("Sorry, those indices are invalid. Please input integral indices between 0 and 2 inclusive, in the correct format:    "))
		except:
		    print ("I'm sorry, but x and y should be numerical.")
	    
	    state[x,y] = -1
	    reward = getLoss(state)
	    if reward != 0: #Game is done
		end = True
		
    #end While
    
main()