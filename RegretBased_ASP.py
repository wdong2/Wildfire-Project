from CI_util import*
import cvxpy as cp
import numpy as np
import math
import random
from data_processing import*
from scipy.linalg import sqrtm
from scipy.linalg import norm

# the optimization function 4.9-4.11 assuming k=10
def optimize(Pi_l,B,Da,A,pi_t,D_bar,rho=1):
    k = len(A) #it must be 10
    eeT = np.ones((len(Pi_l),len(Pi_l)))
    t = cp.Variable()
    f = cp.Variable((2*k))  
    T = get_T()
    temp0 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[0])
    temp1 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[1])
    temp2 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[2])
    temp3 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[3])
    temp4 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[4])
    temp5 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[5])
    temp6 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[6])
    temp7 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[7])
    temp8 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[8])
    temp9 = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[9])
    objective = cp.Minimize(t+ rho*cp.quad_form(f,D_bar))
    constraints = [t >=  cp.norm(temp0@f.T,2), 
                   t >=  cp.norm(temp1@f.T,2), 
                   t >=  cp.norm(temp2@f.T,2), 
                   t >=  cp.norm(temp3@f.T,2), 
                   t >=  cp.norm(temp4@f.T,2), 
                   t >=  cp.norm(temp5@f.T,2), 
                   t >=  cp.norm(temp6@f.T,2), 
                   t >=  cp.norm(temp7@f.T,2), 
                   t >=  cp.norm(temp8@f.T,2), 
                   t >=  cp.norm(temp9@f.T,2), 
                   A@f == T.T@pi_t, f[9] == 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.SCS)       
    return result, t.value, f.value

# get Pi_l matrix based on pi_l; Pi_l has (k-1)*(k-1) dimension
def get_Pi_l(pi_l):
    Pi_l = np.zeros((len(pi_l)-1,len(pi_l)-1))
    for i in range(len(pi_l)-1):
        Pi_l[i][i] = pi_l[i]
    return Pi_l

# get matrix B based on pi_l; B has 2k*(k-1) dimension
def get_B(pi_l):
    B = np.zeros((2*len(pi_l),len(pi_l)-1))
    for i in range (len(pi_l)-1):
        B[i][i] = 1
        B[i+len(pi_l)][i] = 1
    for i in range (len(pi_l)-1):
        B[-1][i] = -pi_l[i]/pi_l[-1]
    return B

# get matrix Da based on pi_l and p_astar; Da has 2k*2k dimension
def get_Da(pi_l, p_astar):
    k = len(pi_l)
    Da = np.zeros((k,2*k,2*k))
    for a in range (k):
        pa = p_astar[a]
        for i in range(k-1):
            Da[a][i][i] = sum(pa[i+1:])*pi_l[i]
            Da[a][k+i][k+i] = sum(pa[:i+1])*pi_l[i]
        Da[a][2*k-1][2*k-1] = pi_l[-1]    
    return Da

def get_D_bar(pi_l):
    k = len(pi_l)
    D_bar = np.zeros((2*k,2*k))
    e = np.ones(k)
    for i in range(k-1):
        D_bar[i][i] = sum(e[i+1:])*pi_l[i]
        D_bar[k+i][k+i] = sum(e[:i+1])*pi_l[i]
    D_bar[2*k-1][2*k-1] = sum(e)*pi_l[k-1]
    return D_bar

# get f vector based on logging, target policies and p_astar; length: 2k
def get_f_vec(l,t,p,k=10):
    f = np.zeros(2*k)
    for i in range(k):
        f[i] = np.float(off_policy_target_value(i, 0, l, t, p))
        f[i+k] = np.float(off_policy_target_value(i, 1, l, t, p))
    return f

# get matrix A based on pi_l; A has k*2k dimension
def get_A_matrix(pi_l):
    k = len(pi_l)
    A = np.zeros((k,2*k))
    for i in range(k):
        for j in range(k):
            if j<i:
                A[i][j] = pi_l[j]
            else:
                A[i][j+k] = pi_l[j]
    return A
    
# build Pi_l, B, Da and A matrix based on pi_l and p_astar                    
def build_variables(pi_l,p_astar):
    Pi_l = get_Pi_l(pi_l)
    B = get_B(pi_l)
    Da = get_Da(pi_l,p_astar)
    A = get_A_matrix(pi_l)
    D_bar = get_D_bar(pi_l)
    return Pi_l, B, Da, A, D_bar

# running experiment with sepecific runs
def run_experiment_ASP(rho = 1, p_astar = None, runs=1):
    pi_l_list, pi_t_list = get_pil_pit()
    k = len(pi_l_list)
    if p_astar == None:
        p_astar = np.identity(k)
    t_list = []
    f0_list = []
    f1_list = []
    for i in range (runs):
        #print("================RUN "+str(i)+"=================")
        for l in range(k):
            t_list.append([])
            f0_list.append([])
            f1_list.append([])
            for t in range(k):
                #print("====================l,t,p: ",l,t,p,"====================")
                pi_t = pi_t_list[t]
                Pi_l,B,Da,A,D_bar = build_variables(pi_l_list[l],p_astar)
                #print("Pi_l = ",Pi_l, "p_astar = ", "B = ",B,"Da = ",Da,"A = ",A, "D_bar = ", D_bar)
                result, t_value, f_value = optimize(Pi_l,B,Da,A,pi_t,D_bar,rho)
                #print(result, t_value)
                t_list[l].append(t_value)
                f0_list[l].append(f_value[:len(f_value)//2])
                f1_list[l].append(f_value[len(f_value)//2:])
                print(f0_list[l],f1_list[l])
    t_list = np.array(t_list).astype(np.float)
    #saveF(t_list,"T_ASP_Minmax_R.pkl")
    saveF([f0_list,f1_list],"F_0_1_MMR.pkl")
    for l in range(k):
        print("pi_l =",l)
        for t in range(k):
            print(np.round(t_list[l][t],4),end=" ")
        print("")
                    

def compute_t_given_f(temp, f):
    return norm(np.dot(temp,f.T),2)

def compute_max_t(Pi_l,B,Da,A,pi_t,D_bar,f):
    k = len(A) 
    eeT = np.ones((len(Pi_l),len(Pi_l)))    
    t_list = []
    for i in range(len(Da)):
        temp = np.dot(np.dot(sqrtm(np.linalg.inv(Pi_l) - eeT),B.T),Da[i])
        t = compute_t_given_f(temp, f)
        t_list.append(t)
    #print("t_list",t_list)
    return max(t_list)

def run_calculate_t(fname = "F_0_1_optVar.pkl", p_astar = None):
    pi_l_list, pi_t_list = get_pil_pit()
    k = len(pi_l_list)    
    f1 = None
    f0 = None
    if fname == "F_IS.pkl":
        f1 = loadF(fname)
        f0 = np.zeros((len(f1),len(f1[0]),len(f1[0][0])))
        print("f11",f1[1][8])
        if np.dot(pi_l_list[0],f1[0][0])==1:
            print("true")
        else:
            print(np.dot(pi_l_list[0],f1[0][0]))
    else:
        f0,f1 = loadF(fname)
    t_list = []
    if p_astar == None:
        p_astar = np.identity(k)    
    for l in range(k):
        t_list.append([])
        for t in range(k):
            f = np.concatenate((f0[l][t],f1[l][t]),axis=None)
            pi_t = pi_t_list[t]
            Pi_l,B,Da,A,D_bar = build_variables(pi_l_list[l],p_astar)    
            t = compute_max_t(Pi_l,B,Da,A,pi_t,D_bar,f)
            t_list[l].append(t)
    print("t value for estimator: "+fname)
    t_list = np.array(t_list)
    saveF(t_list,"T_ASP_IS.pkl")
    for l in range(k):
        print("pi_l =",l)
        for t in range(k):
            print(np.round(t_list[l][t],4),end=" ")
        print("")    

def calculate_diff(fname1="T_ASP_Minmax_R.pkl", fname2="T_ASP_Minmax.pkl"):
    t1 = np.array(loadF(fname1))
    t2 = np.array(loadF(fname2))
    t_list = t1-t2
    k = len(t_list)
    print("t value of "+fname1+" - "+fname2)
    for l in range(k):
        print("pi_l =",l)
        for t in range(k):
            print(np.round(t_list[l][t],4),end=" " )
        print("")   
    flat_t_list = t_list.flatten()
    print("max:",max(flat_t_list), " mean:", np.mean(flat_t_list), " min:",min(flat_t_list))
    

def main():
    #rho = 0.0001
    #run_experiment_ASP(rho)
    #run_calculate_t("F_IS.pkl")
    #calculate_diff("T_ASP_Minmax_R.pkl", "T_ASP_IS.pkl")
    calculate_diff("T_ASP_RIS.pkl", "T_ASP_IS.pkl")
    
main()
            
    
    
