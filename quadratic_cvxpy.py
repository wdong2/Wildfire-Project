import cvxpy as cp
import numpy as np
import math
import data_processing as dp
import os

#-----------------------------------------------------------------------
#  Objective:
#  minimize    alpha^2 + mu
#  subject to  mu*e' >= pi_l' * F^(2) - 2alpha*pi_t'*T (equation 7)
#              pi_l' * F = pi_t' * T (equation 8)
#  k = 10
#  T_a,a* = [[1,0,0,0,0,0,0,0,0,0][1,1,0,0,0...]...] (m.addMVar)
#  a,a* : [1:k]
#  e : [1,1,1,..] (can ignore)
#  make F to be k one dimentional Mvar
#-----------------------------------------------------------------------

# fix p option
#fix_p = True
#index_p = 0
#print("index_p:",index_p)
#p_file = dp.loadF("h1.pkl")
#p_fix = np.array(p_file[index_p]).astype(np.float)
#p_fix = [1,0,0,0,0,0,0,0,0,0]


#=======================================================================================
# Function definition
def worst_p_star(pi_l, pi_t, F, T):
    '''
    calculate the worst p star value from(4)
    '''
    k = len(pi_l)
    p = cp.Variable(k)
    e = np.ones(k)
    objective = cp.Minimize(cp.square(pi_t @ T @ p) - pi_l @ cp.square(F) @ p)
    constraints = [e @ p == 1, p>=0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    p_return = p.value
    return result, p_return

def calculate_variance(pi_l, pi_t, p, F, T, fix_p = False, p_fix = None):
    '''
    calculate the variance based on the worst p star value
    '''
    if fix_p == True:
        p = p_fix
    
    result = np.dot(pi_l , np.dot(np.square(F) , p)) - np.square(np.dot(pi_t , np.dot(T , p)))
    
    return result


def optimize(imp_sam = False, opt_mode = "variance", print_detail = False, fix_p = False, p_fix = None, id=""):
    ''' 
    this is the main function to get optimized variance based on mode and method
    '''
    # store results
    F_a0 = []
    F_a1 = []
    F_IS = []
    worst_p_star_list = []
    variance_list = []
    
    # get pi_l and pi_t from Mostafa
    k = 10
    #file_name = "df_sim_k10_h1.csv"
    #file_list = dp.readcsv(file_name)[1:]
    #pi_l_list = []
    #pi_t_list = []
    #for l in range(k+1):
        #pi_l_list.append([])
        #pi_t_list.append([])
        #for i in range(k):
            #pi_l_list[l].append(file_list[l*110+l*10+i][3])
            #pi_t_list[l].append(file_list[l*110+l*10+i][4])
    #pi_l_list = np.array(pi_l_list).astype(np.float)
    #pi_t_list = np.array(pi_t_list).astype(np.float)
    pi_l_list, pi_t_list = dp.get_pil_pit()
    
    # init variables
    T = np.zeros((k,k))
    e = cp.Parameter(k)
    e.value = np.ones(k)
    for i in range(k):
        for j in range(k):
            if i>=j:
                T[i][j] = 1
    if print_detail == True:
        print("T:",T)
    
    # Construct the problem for f --------------------------------------------------------
    if not imp_sam:
        for l in range(k+1):
            F_a0.append([])
            F_a1.append([])
            variance_list.append([])
            worst_p_star_list.append([])
            for t in range(k+1):
                pi_l = pi_l_list[l]
                pi_t = pi_t_list[t]
                alpha = cp.Variable()
                mu = cp.Variable()
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
                objective = None
                constraints = None
                # set obj and constraints based on the optimization target
                if opt_mode == "variance":
                    objective = cp.Minimize(cp.square(alpha) + mu)
                    constraints = [mu*e >= pi_l @ cp.square(F) - 2*alpha*pi_t@T, pi_l@ F == pi_t@ T, 
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
                elif opt_mode == "regret":
                    # calculate S_sq
                    S_sq = pi_t/pi_l
                    T_copy = np.zeros((k,k))
                    for i in range(k):
                        for j in range(k):
                            if i>=j:
                                T_copy[i][j] = 1          
                    for i in range(len(T)):
                        T_copy[i] *= S_sq[i]
                    S_sq = cp.square(T_copy)
                    # setting obj and constraints
                    objective = cp.Minimize(mu)
                    constraints = [mu*e >= pi_l @ cp.square(F) - pi_l @ S_sq, pi_l @ F == pi_t@ T,
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
        
                # The optimal objective value is returned by `prob.solve()`.
                result = prob.solve()
                if print_detail == True:
                    print("--------------------------------------------------------------------")
                    print("pi_l =",l,"pi_t = ",t)
                    print("total problem result (min of alpha^2 + mu):", result)
                    # The optimal value for any var is stored in `var.value`.
                    print("alpha:", alpha.value)
                    print("mu:", mu.value)
                    print("F:", F.value)
        
                # get worst p_star and variance
                wps = worst_p_star(pi_l, pi_t, F.value, T)
                p = wps[1]
                variance = calculate_variance(pi_l, pi_t, p, F.value, T, fix_p, p_fix)
                if print_detail == True:
                    print("warst p star: ", wps[0], "p: ", wps[1])
                    print("variance:", variance)
                
                # store variable for plot
                F_a0[l].append([F10.value,F20.value,F30.value,F40.value,F50.value,F60.value,F70.value,F80.value,F90.value,0])
                F_a1[l].append([F11.value,F21.value,F31.value,F41.value,F51.value,F61.value,F71.value,F81.value,F91.value,F101.value])
                worst_p_star_list[l].append(p)
                variance_list[l].append(variance)
                
    # Construct the problem for importance sampling --------------------------------------------------------
    else:
        for l in range(k+1):
            F_IS.append([])
            worst_p_star_list.append([])
            variance_list.append([])
            for t in range(k+1):
                pi_l = pi_l_list[l]
                pi_t = pi_t_list[t]
                F = []
                for fi in range(len(T)):
                    F.append(T[fi] * (pi_t/pi_l)[fi])
                F = np.array(F).astype(np.float)
                # get worst p_star and variance
                wps = worst_p_star(pi_l, pi_t, F, T)
                p = wps[1]
                variance = calculate_variance(pi_l, pi_t, p, F, T, fix_p, p_fix)          
                if print_detail == True:
                    print("--------------------------------------------------------------------")
                    print("pi_l =",l,"pi_t = ",t)
                    print("warst p star: ", wps[0], "p: ", wps[1])
                    print("variance:", variance)         
                    print("F:", F)
                # store variable for plot
                F_IS[l].append(F.T[0])   
                worst_p_star_list[l].append(p)     
                variance_list[l].append(variance)
    
    # save variables into .pkl file for plots
    dirname = os.path.dirname(__file__)
    fpath = os.path.join(dirname, 'process_data/')    
    if not imp_sam:
        F_a0 = np.array(F_a0).astype(np.float)
        F_a1 = np.array(F_a1).astype(np.float)
        worst_p_star_list = np.array(worst_p_star_list).astype(np.float)
        if opt_mode == "variance":
            dp.saveF([F_a0,F_a1], fpath + "F_0_1_optVar" + id + ".pkl")
            dp.saveF(worst_p_star_list, fpath + "worst_p_star_list_f_optVar" + id + ".pkl")
            dp.saveF(variance_list, fpath + "var_f_optVar" + id + ".pkl")
            print("F_a0 and F_a1 => F_0_1_optVar.pkl, worst_p_star_list => worst_p_star_list_f_optVar.pkl, variance => var_f_optVar.pkl" + fpath)
        elif opt_mode == "regret":
            dp.saveF([F_a0,F_a1], fpath + "F_0_1_optReg" + id + ".pkl")
            dp.saveF(worst_p_star_list, fpath + "worst_p_star_list_f_optReg" + id + ".pkl")   
            dp.saveF(variance_list, fpath + "var_f_optReg" + id + ".pkl")
            print("F_a0 and F_a1 => F_0_1_optReg.pkl, worst_p_star_list => worst_p_star_list_f_optReg.pkl, variance => var_f_optReg.pkl"+ fpath)            
            
    else:
        F_IS = np.array(F_IS).astype(np.float)
        print('F_IS',F_IS)
        worst_p_star_list = np.array(worst_p_star_list).astype(np.float)
        #dp.saveF(F_IS,  fpath + "F_IS" + id + ".pkl")
        #dp.saveF(worst_p_star_list, fpath + "worst_p_star_list_IS" + id + ".pkl")
        #dp.saveF(variance_list, fpath + "var_IS" + id + ".pkl")
        print("F_IS => F_IS.pkl, worst_p_star_list is saved in worst_p_star_list_IS.pkl file, variance => var_IS.pkl" + fpath)
    
    # The optimal Lagrange multiplier for a constraint is stored in
    # `constraint.dual_value`.
    #print(constraints[0].dual_value)

#=======================================================================================

def main():
    
    # whether this is for importance sampling or f method
    imp_sam = False
    # optimization mode ("variance" or "regret"), only for f method not IS
    opt_mode = "regret"
    # the bool var of whether print all the detail
    print_detail = False    
    
    # get result
    #optimize(imp_sam, opt_mode, print_detail)
    
    optimize(True,None,True)
    
    #fix_p = True
    #index_p = 0
    #p_file = dp.loadF("h1.pkl")
    #p_fix = 0       
    #for i in range(10):
        #index_p = i
        #p_fix = np.array(p_file[index_p]).astype(np.float)        
        #optimize(False, "variance", False, True, p_fix, str(i))
    
    #if fix_p == True:
        #print("fix p_star:",p_fix)
    
    print("Done!")

main()