import cvxpy as cp
import numpy as np
import math
import random
import os
import data_processing as dp
import scipy.stats
from scipy.stats import f
from scipy.stats import chi2

# lower triangle matrix
def get_T(k=10):
    T = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i>=j:
                T[i][j] = 1    
    return T

# value of the policy
def policy_value(policy, p_astar):
    return np.dot(np.dot(policy, get_T(len(policy))), p_astar)

# get index of the norm value
def get_index_norm(norm_value, lst):
    for i in range (len(lst)):
        if norm_value < lst[i]:
            return i
        norm_value -= lst[i]
        
# get a list of the index of the norm value
def get_list_index_norm(norm_value_lst, lst):
    n = len(norm_value_lst)
    return_list = []
    for i in range(n):
        return_list.append(get_index_norm(norm_value_lst[i], lst))
    return return_list

# get ramdom astar and action, calculate delta (a >= astar)
def get_delta(n, p_astar, pi_l, seed = 1):
    np.random.seed(seed)
    astar = np.random.random(n)
    a = np.random.random(n)
    astar = get_list_index_norm(astar, p_astar)
    a = get_list_index_norm(a, pi_l)
    delta = []
    for i in range(n):
        if a[i] >= astar[i]:
            delta.append([a[i],1])
        else:
            delta.append([a[i],0])
    return delta

## get off policy value f(a, delta)
#def off_policy_target_value(a, delta, f0_vec, f1_vec):
    #if delta == 0:
        #return f0_vec[a]
    #else:
        #return f1_vec[a]
    
# get off policy value by a, delta, index of pi_l, pi_t, p_astar
def off_policy_target_value(a, delta, pi_l, pi_t, p_astar):
    data = dp.loadF("process_data\F_0_1_optVar"+str(p_astar)+".pkl")
    data = data[delta][pi_l][pi_t][a]
    return data

# get a list of policy value by (a,delta) "ins", index of pi_l, pi_t and p_astar
def off_policy_target_values(ins, pi_l, pi_t, p_astar):
    return_list = []
    for i in range(len(ins)):
        data = off_policy_target_value(ins[i][0], ins[i][1], pi_l, pi_t, p_astar)
        return_list.append(data)
    return return_list
    
# solve CI upper bound
def solve_w_max (policy_value, r0):
    n = len(policy_value)
    w = cp.Variable(n)
    objective = cp.Maximize(policy_value @ w)
    constraints = [w>=0, sum(w)==1, -cp.sum(cp.log(n*w)) <= -cp.log(r0)]   
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS)    
    if result == None:
        print("None found CI_util.py line 84")
        result = 1000      
    w_v = w.value
    return result, w_v

# solve CI lower bound
def solve_w_min (policy_value, r0):
    n = len(policy_value)
    w = cp.Variable(n)
    objective = cp.Minimize(policy_value @ w)
    constraints = [w>=0, sum(w)==1, -cp.sum(cp.log(n*w)) <= -cp.log(r0)]  
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS)    
    if result == None:
        print("None found CI_util.py line 84")
        result = -1000      
    w_v = w.value
    return result, w_v

# get r0 depends on method
def get_r0_f(is_f, n, alpha = 0.05):
    if is_f:
        Q = f.ppf(1-alpha, 1, n-1)
    else:
        Q = chi2.ppf(1-alpha, df=1)
    return math.exp(-Q/2)

# make data set based on # of samples, # of runes, pi_l, pi_t
def make_data_set(n, runs, pi_l, pi_t, p_astar):
    data_set = []
    for i in range(runes):
        delta = get_delta(n, p_astar, pi_l, i)
        data_set.append(delta)
    return data_set

# calculate CI based on central limit theorem
def CI_CLT(data, alpha = 0.05):
    n = len(data)
    m, sigma = np.mean(data), np.std(data)
    l, h = scipy.stats.norm.interval(1-0.05, m, sigma/math.sqrt(n))
    return  l, h

# calculate CI based on t-student distribution
def CI_T(data, alpha = 0.05):
    n = len(data)
    m, sigma = np.mean(data), np.std(data)
    l , h = scipy.stats.t.interval(1-alpha, n-1, m, sigma/math.sqrt(n))
    return l, h

# get variance info from index
def get_variance_info(l, t, p):
    var_list = loadF("process_data\var_f_optVar"+str(p)+".pkl")
    return var_list[l,t]
    