#!/usr/bin/env python3.7

#  Objective:
#  maximize    alpha^2 + mu
#  subject to  mu*e' >= pi_l' * F^(2) - 2alpha*pi_t'*T (7)
#              pi_l' * F = pi_t' * T (8)
#  k = 10
#  T_a,a* = [[1,0,0,0,0,0,0,0,0,0][1,1,0,0,0...]...] (m.addMVar)
#  a,a* : [1:k]
#  e : [1,1,1,..] (can ignore)
#  make F to be k one dimentional Mvar

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB

# Create constent
k = 10
T = np.zeros((k,k))
e = np.ones(k)
for i in range(k):
    for j in range(k):
        if i>=j:
            T[i][j] = 1
pi_l = np.zeros(k)
pi_t = np.zeros(k)
policy = [0,0]
pi_l[policy[0]] = 1
pi_t[policy[1]] = 1
# transfer T
T = T.T

# Create a new model
m = gp.Model("qcp")

# Create variables
alpha = m.addMVar(1,name="alpha")
alpha_s = m.addMVar(1,name="alpha_s")
mu = m.addMVar(1,name="mu")
# vertical vectors
if k >= 10: 
    F1 = m.addMVar(k,name="F1")
    F2 = m.addMVar(k,name="F2")
    F3 = m.addMVar(k,name="F3")
    F4 = m.addMVar(k,name="F4")
    F5 = m.addMVar(k,name="F5")
    F6 = m.addMVar(k,name="F6")
    F7 = m.addMVar(k,name="F7")
    F8 = m.addMVar(k,name="F8")
    F9 = m.addMVar(k,name="F9")
    F10 = m.addMVar(k,name="F10")
    F1_s = m.addMVar(k,name="F1_s")
    F2_s = m.addMVar(k,name="F2_s")
    F3_s = m.addMVar(k,name="F3_s")
    F4_s = m.addMVar(k,name="F4_s")
    F5_s = m.addMVar(k,name="F5_s")
    F6_s = m.addMVar(k,name="F6_s")
    F7_s = m.addMVar(k,name="F7_s")
    F8_s = m.addMVar(k,name="F8_s")
    F9_s = m.addMVar(k,name="F9_s")
    F10_s = m.addMVar(k,name="F10_s")    
#F = np.ones((k,k))

# Set objective
obj = alpha_s + mu
m.setObjective(obj, GRB.MAXIMIZE)

# Add constraint
# Add constrain (7)
for i in range(k):
    m.addConstr(F1_s[i]==F1[i]@F1[i])
for i in range(k):
    m.addConstr(F2_s[i]==F2[i]@F2[i])
for i in range(k):
    m.addConstr(F3_s[i]==F3[i]@F3[i])
for i in range(k):
    m.addConstr(F4_s[i]==F4[i]@F4[i])
for i in range(k):
    m.addConstr(F5_s[i]==F5[i]@F5[i])
for i in range(k):
    m.addConstr(F6_s[i]==F6[i]@F6[i])
for i in range(k):
    m.addConstr(F7_s[i]==F7[i]@F7[i])
for i in range(k):
    m.addConstr(F8_s[i]==F8[i]@F8[i])
for i in range(k):
    m.addConstr(F9_s[i]==F9[i]@F9[i])
for i in range(k):
    m.addConstr(F10_s[i]==F10[i]@F10[i])                        
m.addConstr(alpha_s[0]==alpha[0]@alpha[0]) 

# Add constrain (7)
m.addConstr(mu >= pi_l@F1_s - 2*alpha*(pi_t@T[0]))
m.addConstr(mu >= pi_l@F2_s - 2*alpha*(pi_t@T[1]))
m.addConstr(mu >= pi_l@F3_s - 2*alpha*(pi_t@T[2]))
m.addConstr(mu >= pi_l@F4_s - 2*alpha*(pi_t@T[3]))
m.addConstr(mu >= pi_l@F5_s - 2*alpha*(pi_t@T[4]))
m.addConstr(mu >= pi_l@F6_s - 2*alpha*(pi_t@T[5]))
m.addConstr(mu >= pi_l@F7_s - 2*alpha*(pi_t@T[6]))
m.addConstr(mu >= pi_l@F8_s - 2*alpha*(pi_t@T[7]))
m.addConstr(mu >= pi_l@F9_s - 2*alpha*(pi_t@T[8]))
m.addConstr(mu >= pi_l@F10_s - 2*alpha*(pi_t@T[9]))

# Add constrain (8)
m.addConstr(pi_l@F1 == pi_t@T[0])
m.addConstr(pi_l@F2 == pi_t@T[1])
m.addConstr(pi_l@F3 == pi_t@T[2])
m.addConstr(pi_l@F4 == pi_t@T[3])
m.addConstr(pi_l@F5 == pi_t@T[4])
m.addConstr(pi_l@F6 == pi_t@T[5])
m.addConstr(pi_l@F7 == pi_t@T[6])
m.addConstr(pi_l@F8 == pi_t@T[7])
m.addConstr(pi_l@F9 == pi_t@T[8])
m.addConstr(pi_l@F10 == pi_t@T[9])

m.optimize()


