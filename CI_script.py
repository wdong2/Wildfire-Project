import cvxpy as cp
import numpy as np
import math
import random
from data_processing import*
import CI_util as CI

def run_CI(runs, n_ins, methods, astar_file):
    n_pi_l = 11
    n_pi_t = 11
    n_astar = 10
    p_file = loadF(astar_file)
    pi_l_list, pi_t_list = get_pil_pit()
    r0_f = CI.get_r0_f(True, n_ins)
    r0_chi = CI.get_r0_f(False, 0)   
    true_value_list = []
    target_vals_list = []
    clt_inrange = []
    t_inrange = []
    chi_inrange = []
    f_inrange = []
    temp_seed = [180,181,182]
    for i in range(runs):
        print("run",i)
        for j in range(n_pi_l):
            pi_l = pi_l_list[j]
            for k in range(n_astar):
                p_astar = np.array(p_file[k]).astype(np.float) 
                for l in range(n_pi_t):
                    pi_t = pi_t_list[l]
                    ins = CI.get_delta(n_ins, p_astar, pi_l, temp_seed[i]) 
                    true_value = CI.policy_value(pi_t, p_astar)
                    true_value_list.append(true_value)
                    target_vals = CI.off_policy_target_values(ins, j, l, k)
                    target_vals_list.append(np.mean(target_vals))
                    # CI CLT
                    clt_low, clt_high = CI.CI_CLT(target_vals)
                    # CI t
                    t_low, t_high = CI.CI_T(target_vals)
                    # CI chi-square
                    chi_low = CI.solve_w_min(target_vals, r0_chi)
                    chi_high = CI.solve_w_max(target_vals, r0_chi)
                    # CI f
                    f_low = CI.solve_w_min(target_vals, r0_f)
                    f_high = CI.solve_w_max(target_vals, r0_f)
                    
                    # get coverage info
                    #print (clt_low, true_value, clt_high)
                    if true_value >= clt_low and true_value <= clt_high:
                        clt_inrange.append(1)
                    else:
                        clt_inrange.append(0)
                        
                    if true_value >= t_low and true_value <= t_high:
                        t_inrange.append(1)
                    else:
                        t_inrange.append(0)    
    
                    if true_value >= chi_low[0] and true_value <= chi_high[0]:
                        chi_inrange.append(1)
                    else:
                        chi_inrange.append(0)  
                    
                    if true_value >= f_low[0] and true_value <= f_high[0]:
                        f_inrange.append(1)
                    else:
                        f_inrange.append(0)      
    
    print("coverage CLT:", sum(clt_inrange)/len(clt_inrange))
    print("coverage T:", sum(t_inrange)/len(t_inrange))
    print("coverage Chi2:", sum(chi_inrange)/len(chi_inrange))
    print("coverage f:", sum(f_inrange)/len(f_inrange))
    saveF([true_value_list,target_vals_list,clt_inrange,t_inrange,chi_inrange,f_inrange], str(runs)+"r_CI_result.pkl")
    
def main():
    # runs, n_ins, methods, astar_file
    runs = 3
    run_CI(runs,100,None,"h1.pkl")
    
main()