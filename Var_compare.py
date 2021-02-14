import numpy as np
import math
import data_processing as dp
import os

#calculate the variance based on formula
def calculate_variance(pi_l, pi_t, p, F):
    T = dp.get_T()
    part1 = np.dot(pi_l , np.dot(np.square(F) , p))
    part2 = np.square(np.dot(pi_t , np.dot(T , p)))
    result = part1 - part2
    return result

# transform f0 and f1 vector to F matrix
def change_f01_to_F(f0,f1):
    print("f0,f1",f0,f1)
    F = np.zeros((len(f0),len(f0)))
    for i in range (len(f0)):
        for j in range (len(f0)):
            if j > i:
                F[i][j] = f0[i]
            else:
                F[i][j] = f1[i]
    print("F",F)
    return F

def calculate_variance_h1(f_fname="F_0_1_MMR.pkl"):
    pi_l_list, pi_t_list = dp.get_pil_pit()
    pi_l_list = np.array(pi_l_list).astype(np.float)
    pi_t_list = np.array(pi_t_list).astype(np.float)
    k = len(pi_l_list[0])
    p_list = dp.loadF("h1.pkl")[:-1]
    p_list = np.array(p_list).astype(np.float)
    print("p_list",p_list)
    f0_list, f1_list = dp.loadF(f_fname)
    #print("0,1",f0_list[0],f1_list[0])
    Var_list = []
    for l in range(len(pi_l_list)):
        Var_list.append([])
        for t in range(len(pi_t_list)):
            Var_list[l].append([])
            F = change_f01_to_F(f0_list[l][t],f1_list[l][t])
            for p in range(k):
                var = calculate_variance(pi_l_list[l], pi_t_list[t], p_list[p], F)
                Var_list[l][t].append(var)
    dp.saveF(Var_list,"var_ltp_h1_"+f_fname[6:-4])
    print("var_list",Var_list)
    
def main():
    calculate_variance_h1("F_0_1_MMR.pkl")
    
main()