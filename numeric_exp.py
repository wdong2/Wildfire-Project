import cvxpy as cp
import numpy as np
import math
import data_processing as dp
import os

    
def best_estimater_F(l,t,p,k=10):
    pi_l = l
    pi_t = t
    T = dp.get_T()
    F11 = cp.Variable()
    F10 = cp.Variable()
    F21 = cp.Variable()
    F20 = cp.Variable()
    F31 = cp.Variable()
    F30 = cp.Variable()
    F41 = cp.Variable()
    F40 = cp.Variable()
    F51 = cp.Variable()
    F50 = cp.Variable()
    F61 = cp.Variable()
    F60 = cp.Variable()
    F71 = cp.Variable()
    F70 = cp.Variable()
    F81 = cp.Variable()
    F80 = cp.Variable()
    F91 = cp.Variable()
    F90 = cp.Variable()
    F101 = cp.Variable()
    F = cp.Variable((k,k))    
    objective = cp.Minimize(pi_l @ cp.square(F) @ p - cp.square(pi_t @ T @ p))
    constraints = [pi_l@ F == pi_t@ T, 
                   F[0][0]==F11, F[0][1:10]==F10,
                   F[1][0:2]==F21, F[1][2:10]==F20,
                   F[2][0:3]==F31, F[2][3:10]==F30,
                   F[3][0:4]==F41, F[3][4:10]==F40,
                   F[4][0:5]==F51, F[4][5:10]==F50,
                   F[5][0:6]==F61, F[5][6:10]==F60,
                   F[6][0:7]==F71, F[6][7:10]==F70,
                   F[7][0:8]==F81, F[7][8:10]==F80,
                   F[8][0:9]==F91, F[8][9:10]==F90,
                   F[9][0:10]==F101
                   ]    
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    F_a0 = [F10.value,F20.value,F30.value,F40.value,F50.value,F60.value,F70.value,F80.value,F90.value,0]
    F_a1 = [F11.value,F21.value,F31.value,F41.value,F51.value,F61.value,F71.value,F81.value,F91.value,F101.value]
    
    return_F = np.array([F_a0,F_a1]).astype(np.float)
    return [F.value, return_F, result]

def pareto_optimal(F_bar,l,t,rho=0.001,k=10):
    e = np.ones(k)
    mu = cp.Variable()
    T = dp.get_T()
    F11 = cp.Variable()
    F10 = cp.Variable()
    F21 = cp.Variable()
    F20 = cp.Variable()
    F31 = cp.Variable()
    F30 = cp.Variable()
    F41 = cp.Variable()
    F40 = cp.Variable()
    F51 = cp.Variable()
    F50 = cp.Variable()
    F61 = cp.Variable()
    F60 = cp.Variable()
    F71 = cp.Variable()
    F70 = cp.Variable()
    F81 = cp.Variable()
    F80 = cp.Variable()
    F91 = cp.Variable()
    F90 = cp.Variable()
    F101 = cp.Variable()
    F = cp.Variable((k,k))    
    objective = cp.Minimize(rho*l@cp.square(F)@e + mu)
    constraints = [mu*e >= l @ cp.square(F) - l @ cp.square(F_bar), 
                   l@F == t@T,
                   F[0][0]==F11, F[0][1:10]==F10,
                   F[1][0:2]==F21, F[1][2:10]==F20,
                   F[2][0:3]==F31, F[2][3:10]==F30,
                   F[3][0:4]==F41, F[3][4:10]==F40,
                   F[4][0:5]==F51, F[4][5:10]==F50,
                   F[5][0:6]==F61, F[5][6:10]==F60,
                   F[6][0:7]==F71, F[6][7:10]==F70,
                   F[7][0:8]==F81, F[7][8:10]==F80,
                   F[8][0:9]==F91, F[8][9:10]==F90,
                   F[9][0:10]==F101                   
                   ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS)    
    return result
    

def main():
    F_a0_list = []
    F_a1_list = []
    p_o_list = [] # pareto optimal
    k = 10
    pi_l_list, pi_t_list = dp.get_pil_pit()
    p_astar = [pi_l_list[0]]
    
    for p in range(len(p_astar)):
        F_a0_list.append([])
        F_a1_list.append([])        
        p_o_list.append([])   
        for l in range(k+1):
            F_a0_list[p].append([])
            F_a1_list[p].append([])    
            p_o_list[p].append([])   
            for t in range(k+1):
                F,[F_a0, F_a1], temp_value = best_estimater_F(pi_l_list[l], pi_t_list[t], p_astar[p])
                F_a0_list[p][l].append(F_a0)
                F_a1_list[p][l].append(F_a1)
                #print(F,F_a0,F_a1)
                p_o = pareto_optimal(F,pi_l_list[l], pi_t_list[t])
                p_o_list[p][l].append(p_o)
                print("t",t)
            print("l",l)
        print("p",p)
    
    F_a0_list = np.array(F_a0_list).astype(np.float)
    F_a1_list = np.array(F_a1_list).astype(np.float)
    print (p_o_list)
    print("Done!")
                
main()