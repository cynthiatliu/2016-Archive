#Running functions in lattice_calculator_procedural2.py and ubmatrix.py
#For better understanding of what's going on

import numpy as np
import math
import lattice_calculator_procedural2 as lcp2
import ubmatrix as ub

def main():
    astar, bstar, cstar, alphastar, betastar, gammastar = ub.star(6, 5, 4, 60, 90, 90)
    recip_lp = [astar, bstar, cstar, alphastar, betastar, gammastar]
    print (recip_lp)
    
    B = ub.calcB(astar, bstar, cstar, alphastar, betastar, gammastar, 4, 60)
    print ("---------------------------------")
    print (np.array_str(B))
    
    U = ub.calcU(h1, k1, l1, h2, k2, l2, omega1, chi1, phi1, omega2, chi2, 
                phi2, Bmatrix)