#Tic tac toe
#How to implement RL

import numpy as np
# from sknn.mlp import Regressor, Convolution, Layer
# import q_network
# import pyrl.agents.policy_gradient_cl as pg
import math
import random
import sys
import tensorflow as tf

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

    #Variables
    gamma = .3
    tau = .02
    
    #Initialize graph
    #layer 1
    x00 = tf.constant(0.0); x01 = tf.constant(0.0); x02 = tf.constant(0.0)
    x10 = tf.constant(0.0); x11 = tf.constant(0.0); x12 = tf.constant(0.0)
    x20 = tf.constant(0.0); x21 = tf.constant(0.0); x22 = tf.constant(0.0)
    a = tf.constant(0)
    
    #layer 2
    x200 = tf.constant(0.0); x201 = tf.constant(0.0); x202 = tf.constant(0.0)
    x210 = tf.constant(0.0); x211 = tf.constant(0.0); x212 = tf.constant(0.0)
    x220 = tf.constant(0.0); x221 = tf.constant(0.0); x222 = tf.constant(0.0)
    a2 = tf.constant(0)
    
    #output layer
    #???
    
    #First layer weights
    ws1 = np.random.randn(45)
    
    #Second layer weights
    ws2 = np.random.randn(9)
    
    
    