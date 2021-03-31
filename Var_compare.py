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


def calculate_rho(a,b):
    if a>b:
        return (a-b)/a
    else:
        return (a-b)/b

def compare_var(first = "var_ltp_h1_optReg",second = "var_ltp_h1_MMR"):
    first_var_list = dp.loadF(first)
    second_var_list = dp.loadF(second)
    first_vat_list = np.array(first_var_list)
    second_var_list = np.array(second_var_list)
    L = len(first_var_list)
    T = len(first_var_list[0])
    P = len(first_var_list[0][0])
    total_len = L*T*P
    print(total_len)
    # all 10 p
    rho10 = []
    rho25 = []
    rho50 = []
    rho75 = []
    rho_10 = []
    rho_25 = []
    rho_50 = []
    rho_75 = []
    pos = []
    neg = []
    all_rho = []
    L15_T610_rho = []
    L15_T610_diff = []
    L15_T610_pos = 0
    L610_T15_rho = []
    L610_T15_diff = []
    L610_T15_pos = 0
    L15_T15_rho = []
    L15_T15_diff = []
    L15_T15_pos = 0
    L610_T610_rho = []
    L610_T610_diff = []
    L610_T610_pos = 0
    diag_rho = []
    diag_diff = []
    diag_pos = 0
    for l in range(L):
        for t in range(T):
            for p in range(P):
                rho = calculate_rho(first_var_list[l][t][p],second_var_list[l][t][p])
                all_rho.append(rho)
                diff = first_var_list[l][t][p] - second_var_list[l][t][p]
                if rho >= 0:
                    pos.append(diff)
                    if rho >= 0.75:
                        rho75.append(diff)
                    if rho >= 0.5:
                        rho50.append(diff)
                    if rho >= 0.25:
                        rho25.append(diff)
                    if rho >= 0.1:
                        rho10.append(diff)
                else:
                    neg.append(diff)
                    if rho <= -0.75:
                        rho_75.append(diff)
                    if rho <= -0.5:
                        rho_50.append(diff)
                    if rho <= -0.25:
                        rho_25.append(diff)
                    if rho <= -0.1:
                        rho_10.append(diff)         
                if l<5:
                    if t<5:
                        L15_T15_rho.append(rho)
                        L15_T15_diff.append(diff)
                        if rho>=0:
                            L15_T15_pos+=1
                    else:
                        L15_T610_rho.append(rho)
                        L15_T610_diff.append(diff)
                        if rho>=0:
                            L15_T610_pos+=1
                else:
                    if t<5:
                        L610_T15_rho.append(rho)
                        L610_T15_diff.append(diff)
                        if rho>=0:
                            L610_T15_pos+=1                        
                    else:
                        L610_T610_rho.append(rho)
                        L610_T610_diff.append(diff)
                        if rho>=0:
                            L610_T610_pos+=1    
                if l==t:
                    diag_rho.append(rho)
                    diag_diff.append(diff)
                    if rho>=0:
                        diag_pos+=1
    print("10",len(rho10)/total_len, sum(rho10)/len(rho10))
    print("25",len(rho25)/total_len, sum(rho25)/len(rho25))
    print("50",len(rho50)/total_len, sum(rho50)/len(rho50))
    if len(rho75)!=0:
        print("75",len(rho75)/total_len, sum(rho75)/len(rho75))
    print("-10",len(rho_10)/total_len, sum(rho_10)/len(rho_10))
    print("-25",len(rho_25)/total_len, sum(rho_25)/len(rho_25))
    print("-50",len(rho_50)/total_len, sum(rho_50)/len(rho_50))
    print("-75",len(rho_75)/total_len, sum(rho_75)/len(rho_75))
    print("pos",len(pos)/total_len,L15_T610_pos/len(L15_T610_rho),L610_T15_pos/len(L610_T15_rho),L15_T15_pos/len(L15_T15_rho),L610_T610_pos/len(L610_T610_rho),diag_pos/len(diag_rho))
    print("meandiff", (sum(pos)+sum(neg))/total_len, sum(L15_T610_diff)/len(L15_T610_rho),sum(L610_T15_diff)/len(L610_T15_rho),sum(L15_T15_diff)/len(L15_T15_rho),sum(L610_T610_diff)/len(L610_T610_rho),sum(diag_diff)/len(diag_rho))
    print("meanrho", sum(all_rho)/total_len, sum(L15_T610_rho)/len(L15_T610_rho),sum(L610_T15_rho)/len(L610_T15_rho),sum(L15_T15_rho)/len(L15_T15_rho),sum(L610_T610_rho)/len(L610_T610_rho),sum(diag_rho)/len(diag_rho))
    
    # p 1-5
    total_len/=2
    rho10 = []
    rho25 = []
    rho50 = []
    rho75 = []
    rho_10 = []
    rho_25 = []
    rho_50 = []
    rho_75 = []
    pos = []
    neg = []
    all_rho = []
    L15_T610_rho = []
    L15_T610_diff = []
    L15_T610_pos = 0
    L610_T15_rho = []
    L610_T15_diff = []
    L610_T15_pos = 0
    L15_T15_rho = []
    L15_T15_diff = []
    L15_T15_pos = 0
    L610_T610_rho = []
    L610_T610_diff = []
    L610_T610_pos = 0
    diag_rho = []
    diag_diff = []
    diag_pos = 0
    for l in range(L):
        for t in range(T):
            for p in [0,1,2,3,4]:
                rho = calculate_rho(first_var_list[l][t][p],second_var_list[l][t][p])
                all_rho.append(rho)
                diff = first_var_list[l][t][p] - second_var_list[l][t][p]
                if rho >= 0:
                    pos.append(diff)
                    if rho >= 0.75:
                        rho75.append(diff)
                    if rho >= 0.5:
                        rho50.append(diff)
                    if rho >= 0.25:
                        rho25.append(diff)
                    if rho >= 0.1:
                        rho10.append(diff)
                else:
                    neg.append(diff)
                    if rho <= -0.75:
                        rho_75.append(diff)
                    if rho <= -0.5:
                        rho_50.append(diff)
                    if rho <= -0.25:
                        rho_25.append(diff)
                    if rho <= -0.1:
                        rho_10.append(diff)         
                if l<5:
                    if t<5:
                        L15_T15_rho.append(rho)
                        L15_T15_diff.append(diff)
                        if rho>=0:
                            L15_T15_pos+=1
                    else:
                        L15_T610_rho.append(rho)
                        L15_T610_diff.append(diff)
                        if rho>=0:
                            L15_T610_pos+=1
                else:
                    if t<5:
                        L610_T15_rho.append(rho)
                        L610_T15_diff.append(diff)
                        if rho>=0:
                            L610_T15_pos+=1                        
                    else:
                        L610_T610_rho.append(rho)
                        L610_T610_diff.append(diff)
                        if rho>=0:
                            L610_T610_pos+=1    
                if l==t:
                    diag_rho.append(rho)
                    diag_diff.append(diff)
                    if rho>=0:
                        diag_pos+=1
    print("10",len(rho10)/total_len, sum(rho10)/len(rho10))
    print("25",len(rho25)/total_len, sum(rho25)/len(rho25))
    print("50",len(rho50)/total_len, sum(rho50)/len(rho50))
    if len(rho75)!=0:
        print("75",len(rho75)/total_len, sum(rho75)/len(rho75))
    print("-10",len(rho_10)/total_len, sum(rho_10)/len(rho_10))
    print("-25",len(rho_25)/total_len, sum(rho_25)/len(rho_25))
    print("-50",len(rho_50)/total_len, sum(rho_50)/len(rho_50))
    print("-75",len(rho_75)/total_len, sum(rho_75)/len(rho_75))
    print("pos",len(pos)/total_len,L15_T610_pos/len(L15_T610_rho),L610_T15_pos/len(L610_T15_rho),L15_T15_pos/len(L15_T15_rho),L610_T610_pos/len(L610_T610_rho),diag_pos/len(diag_rho))
    print("meandiff", (sum(pos)+sum(neg))/total_len, sum(L15_T610_diff)/len(L15_T610_rho),sum(L610_T15_diff)/len(L610_T15_rho),sum(L15_T15_diff)/len(L15_T15_rho),sum(L610_T610_diff)/len(L610_T610_rho),sum(diag_diff)/len(diag_rho))
    print("meanrho", sum(all_rho)/total_len, sum(L15_T610_rho)/len(L15_T610_rho),sum(L610_T15_rho)/len(L610_T15_rho),sum(L15_T15_rho)/len(L15_T15_rho),sum(L610_T610_rho)/len(L610_T610_rho),sum(diag_rho)/len(diag_rho))
    
    # p 5-10
    rho10 = []
    rho25 = []
    rho50 = []
    rho75 = []
    rho_10 = []
    rho_25 = []
    rho_50 = []
    rho_75 = []
    pos = []
    neg = []
    all_rho = []
    L15_T610_rho = []
    L15_T610_diff = []
    L15_T610_pos = 0
    L610_T15_rho = []
    L610_T15_diff = []
    L610_T15_pos = 0
    L15_T15_rho = []
    L15_T15_diff = []
    L15_T15_pos = 0
    L610_T610_rho = []
    L610_T610_diff = []
    L610_T610_pos = 0
    diag_rho = []
    diag_diff = []
    diag_pos = 0
    for l in range(L):
        for t in range(T):
            for p in [5,6,7,8,9]:
                rho = calculate_rho(first_var_list[l][t][p],second_var_list[l][t][p])
                all_rho.append(rho)
                diff = first_var_list[l][t][p] - second_var_list[l][t][p]
                if rho >= 0:
                    pos.append(diff)
                    if rho >= 0.75:
                        rho75.append(diff)
                    if rho >= 0.5:
                        rho50.append(diff)
                    if rho >= 0.25:
                        rho25.append(diff)
                    if rho >= 0.1:
                        rho10.append(diff)
                else:
                    neg.append(diff)
                    if rho <= -0.75:
                        rho_75.append(diff)
                    if rho <= -0.5:
                        rho_50.append(diff)
                    if rho <= -0.25:
                        rho_25.append(diff)
                    if rho <= -0.1:
                        rho_10.append(diff)         
                if l<5:
                    if t<5:
                        L15_T15_rho.append(rho)
                        L15_T15_diff.append(diff)
                        if rho>=0:
                            L15_T15_pos+=1
                    else:
                        L15_T610_rho.append(rho)
                        L15_T610_diff.append(diff)
                        if rho>=0:
                            L15_T610_pos+=1
                else:
                    if t<5:
                        L610_T15_rho.append(rho)
                        L610_T15_diff.append(diff)
                        if rho>=0:
                            L610_T15_pos+=1                        
                    else:
                        L610_T610_rho.append(rho)
                        L610_T610_diff.append(diff)
                        if rho>=0:
                            L610_T610_pos+=1    
                if l==t:
                    diag_rho.append(rho)
                    diag_diff.append(diff)
                    if rho>=0:
                        diag_pos+=1
    print("10",len(rho10)/total_len, sum(rho10)/len(rho10))
    print("25",len(rho25)/total_len, sum(rho25)/len(rho25))
    print("50",len(rho50)/total_len, sum(rho50)/len(rho50))
    if len(rho75)!=0:
        print("75",len(rho75)/total_len, sum(rho75)/len(rho75))
    print("-10",len(rho_10)/total_len, sum(rho_10)/len(rho_10))
    print("-25",len(rho_25)/total_len, sum(rho_25)/len(rho_25))
    print("-50",len(rho_50)/total_len, sum(rho_50)/len(rho_50))
    print("-75",len(rho_75)/total_len, sum(rho_75)/len(rho_75))
    print("pos",len(pos)/total_len,L15_T610_pos/len(L15_T610_rho),L610_T15_pos/len(L610_T15_rho),L15_T15_pos/len(L15_T15_rho),L610_T610_pos/len(L610_T610_rho),diag_pos/len(diag_rho))
    print("meandiff", (sum(pos)+sum(neg))/total_len, sum(L15_T610_diff)/len(L15_T610_rho),sum(L610_T15_diff)/len(L610_T15_rho),sum(L15_T15_diff)/len(L15_T15_rho),sum(L610_T610_diff)/len(L610_T610_rho),sum(diag_diff)/len(diag_rho))
    print("meanrho", sum(all_rho)/total_len, sum(L15_T610_rho)/len(L15_T610_rho),sum(L610_T15_rho)/len(L610_T15_rho),sum(L15_T15_rho)/len(L15_T15_rho),sum(L610_T610_rho)/len(L610_T610_rho),sum(diag_rho)/len(diag_rho))
    
    return

def main():
    #calculate_variance_h1("F_0_1_optVar.pkl")
    compare_var("var_ltp_h1_optReg","var_ltp_h1_MMR")
    print("Done!")
    
main()