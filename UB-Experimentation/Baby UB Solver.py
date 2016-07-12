#UB Solver Version 0

#We begin with the correct answer on a cubic crystal
#Cubic crystal --> B = I
#Question: is reinforcement learning even reasonable to consider?

#Lumbsen constraints - keep scattering plane as horizontal as possible, minimize the divation of arcs from 0
#We also hope that the program can find these to minimize movement time

#Note: continuous action space, actor-critic model

import numpy as np
import scipy as sp
import time
from sknn.mlp import Regressor, Convolution, Layer
import q_network
import math
import lattice_calculator_procedural2 as lcp2
import ubmatrix as ub

def relearn(critic, actor, Q_prime, mu_prime, transitions):
    #Sample N transitions (s_i, a_i, r_i, s_{i+1}) from R
    #y = []
    #for i in range (N):
    #    y_i = E(r_i + gamma * max_a' (Q(phi_{i+1}, a'_i; theta)))
    #
    #Update critic by minimizing loss L = (1/N) * sum (y_i - Q(s_i, a_i))^2
    #Update actor by using policy gradient (1/N) * sum grad(Q(s_i, actor.predict(s_i)) * grad(actor.predict(s_i))
    #Adjust target network parameters by training one batch
    #Calculate UB
    
    return critic, actor, Q_prime, mu_prime

def actStep(critic, actor, Q_prime, mu_prime, transitions):
    
    return None

def main():
    
    #Prompt user for parameters and reflections
    a = float(input("Please input the length of (or an estimate of the length of) the a axis: "))
    b = float(input("Please input the length of (or an estimate of the length of) the b axis: "))
    c = float(input("Please input the length of (or an estimate of the length of) the c axis: "))
    alpha = float(input("Please input the degree measure of (or an estimate of the degree measure of) alpha: "))
    beta = float(input("Please input an estimate or value for the degree measure of beta: "))
    gamma = float(input("Please input an estimate or value for the degree measure of gamma: "))
    
    h1, k1, l1 = input("Please input a first hkl triple, with h, k, and l separated by commas: ")
    h1 = int(h1); k1 = int(k1); l1 = int(l1)
    print (h1, k1, l1)
    omega1, chi1, phi1 = input("Please input the omega, chi, and phi angles used to find that reflection (again separated by commas): ")
    omega1 = float(omega1); chi1 = float(chi1); phi1 = float(phi1)
    
    h2, k2, l2 = input("Please input a second hkl triple: ")
    h2 = int(h2); k2 = int(k2); l2 = int(l2)
    omega2, chi2, phi2 = input("Please input the omega, chi, and phi angles used to find that reflection: ")
    omega2 = float(omega2); chi2 = float(chi2); phi1 = float(phi2)
    
    #Calculate initial value of UB
    ast, bst, cst, alphast, betast, gammast = ub.star(a, b, c, alpha, beta, gamma) #Calculates reciprocal parameters
    Bmat = ub.calcB(ast, bst, cst, alphast, betast, gammast, c, alpha) #Calculates the initial B matrix
    print (Bmat)
    ub_0 = ub.calcUB(h1, k1, l1, h2, k2, l2, omega1, chi1, phi1, omega2, chi2, phi2, Bmat) #Calculates the initial UB matrix
    
    #This is where our baby program defers from the program proper
    #Feed program real reflection points
    #Program will NOT scan
    reflection_file = open("LaMnO3 reflections.txt", 'r')
    refs = []
    line = reflection_file.readline()
    while line != '':
        lineArr = line.split()
        works = True
        for part in lineArr: #Looks at whether there are letters in the line
            if re.search('[a-zA-Z]', part): works = False
            
        if ('\n' not in lineArr) and len(lineArr) != 0 and works:
            refs.append([int(float(lineArr[0])), int(float(lineArr[1])), int(float(lineArr[2])), float(lineArr[3])]) #hkl, structure factor sqaured
        line = data.readline()
	
	
    #Initalize actor
    try: #Did we already train this network?
	params_act = open("parameters_act.txt", r)
	#stuff
	params_act.close()
	
    except:
	actor = Regressor(layers, warning=None, weights=None, random_state=None, 
	                 learning_rule='sgd', learning_rate=0.01, 
	                 learning_momentum=0.9, regularize=None, 
	                 weight_decay=None, dropout_rate=None, 
	                 batch_size=1, n_iter=None, n_stable=10, 
	                 f_stable=0.001, valid_set=None, 
	                 valid_size=0.0, loss_type=None, 
	                 callback=None, debug=False, verbose=None)
	mu_prime = actor
        
    #Initialize critic
    #Convert to q_network code once done writing psuedocode
    try: #Did we already train this network?
        params_crt = open("parameter_crt.txt", 'r')
	#stuff
	params_crt.close()
        
    except:
	Q = Regressor(
            layers=[Layer("Maxout", name="layer1", units=6, pieces=2), #3 coordinates, structure factor, 1 action, 1 bias
                    Layer("Maxout", name="layer2", units=6, pieces=2),
                    Layer("Maxout", name="layer3", units=6, pieces=2),
                    Layer("Softmax")],
            learning_rate=0.01,
            n_iter=2)
	Q_prime = Q
    
    #Set time counter t to 0, subtract 1 per time step used
    start_time = time.time()
    t = 0    
	
    #DDPG algorithm, adjusted for this program
    #for episode = 1, M do
    #    #Set variables and constants
    #    i = 1; tau = 1/i; R = []
    #
    #    Initialize a random noise process rnp for action exploration
    #    R = [] #Replay buffer
    #    
    #    a_0 = actor(s_0) + rnp
    #    Execute action a_0, results in state s_1
    #    Fit actual structure factor to theoretical structure factor, convert to z-score, convert to accuracy points
    #    Reward r_0 = accuracy points
    #    calculate ub_1
    #
    #    R.append([s_0, a_0, r_0, s_1])
    #    Q, actor, Q_prime, mu_prime = relearn(Q, actor, Q_prime, mu_prime, R)
    #
    #    WHILE ((chi-squared for ub_i, when fit to ub_{i-1}, is smaller than delta) and t <= 6000):
    #        !!   Remove all the _i's before converting to real code  !!
    #
    #        a_i = actor.predict(s_i)
    ##            
    ##        IF a_i is a ROUGH_SCAN
    ##            Perform scan
    ##            Fit actual structure factors to theoretical structure factors, convert to chi-squared
    ##            Q, actor, Q_prime, mu_prime = relearn(Q, actor, Q_prime, mu_prime, R)
    ##
    ##        ELIF a_i is a SCAN
    ##            Perform scan towards chosen location
    ##            Fit actual structure factors to theoretical structure factors, convert to chi-sqaured
    ##            Convert fit quality to accuracy points - chi-squared?
    ##            Q, actor, Q_prime, mu_prime = relearn(Q, actor, Q_prime, mu_prime, R)
    ##
    ##        ELIF a_i is a FIND
    ##            Perform a_i
    ##            Compare actual structure factor to expected structure factor
    ##            Calculate z-score (assume #neutrons > 50), convert to accuracy points
    ##            Q, actor, Q_prime, mu_prime = relearn(Q, actor, Q_prime, mu_prime, R)    
    #
    #        Perform a_i
    #        Compare actual structure factor to expected structure factor
    #        Calculate z-score (assume #neutrons > 50), convert to accuracy points
    #        Q, actor, Q_prime, mu_prime = relearn(Q, actor, Q_prime, mu_prime, R)
    #        
    #        t = time.time() - start.time()
    #        
    #y_i = r_i  #because terminal state
	
    t = time.time() - start_time
    
    #Saving our learned parameters
    params = Q_learner.get_parameters()
    learned_params = open("parameters.txt", 'w')
    for par in params:
        learned_params.write(par)
    
    #Reinforcement learner seeks to maximize both time_reward and acc_reward
    time_reward = -t
    Acc_reward = 5 #Should be recency weighted sum of accuracy points - we expect early actions to have low accuracy

main()