import numpy as np
import math
import random
from scipy import stats
from data_processing import*
from data_processing import loadF

# get variance info from index
def get_variance_info(l, t, p):
    var_list = loadF("process_data\\var_f_optVar"+str(p)+".pkl")
    return var_list[l][t]

# arrange float values into 10 intervals: 0-0.1, 0.1-0.2 ... 0.9-1
def seperate_to_10(data):
    ten_value_list = np.zeros(10)
    for i in range(len(data)):
        if data[i] > 0.9:
            ten_value_list[9]+=1   
        elif data[i] > 0.8:
            ten_value_list[8]+=1   
        elif data[i] > 0.7:
            ten_value_list[7]+=1   
        elif data[i] > 0.6:
            ten_value_list[6]+=1   
        elif data[i] > 0.5:
            ten_value_list[5]+=1   
        elif data[i] > 0.4:
            ten_value_list[4]+=1   
        elif data[i] > 0.3:
            ten_value_list[3]+=1   
        elif data[i] > 0.2:
            ten_value_list[2]+=1   
        elif data[i] > 0.1:
            ten_value_list[1]+=1   
        else:            
            ten_value_list[0]+=1
    return ten_value_list
            
# draw histalgram for the true values
def analyze_value(filename, index=0):
    true_value_list,target_vals_list,clt_inrange,t_inrange,chi_inrange,f_inrange = loadF(filename)
    if index == 0:
        ten_value_list = seperate_to_10(true_value_list)
    ten_value_list /= len(true_value_list)
    plot_hist("Histalgram of true value", 1, [ten_value_list])
    return

# index: (0:clt, 1:t, 2:chi, 3:f)
def analyze_fail(filename):
    true_value_list,target_vals_list,clt_inrange,t_inrange,chi_inrange,f_inrange = loadF(filename)
    method_list = [clt_inrange,t_inrange,chi_inrange,f_inrange]
    target_data_list = []
    for m in range(len(method_list)):
        analyze_method = method_list[m]
        target_data = []
        for i in range(len(analyze_method)):
            if analyze_method[i] == 0:
                target_data.append(true_value_list[i])
        ten_value_list = seperate_to_10(target_data)
        ten_value_list /= len(target_data)
        target_data_list.append(ten_value_list)
    plot_hist("Histalgram of true value not in CI (clt, t, chi, f)", len(method_list), target_data_list, 0, 10, 0, 0.25)
    
# index: (0:clt, 1:t, 2:chi, 3:f)  
def analyze_success(filename):
    true_value_list,target_vals_list,clt_inrange,t_inrange,chi_inrange,f_inrange = loadF(filename)
    method_list = [clt_inrange,t_inrange,chi_inrange,f_inrange]
    target_data_list = []
    for m in range(len(method_list)):
        analyze_method = method_list[m]
        target_data = []
        for i in range(len(analyze_method)):
            if analyze_method[i] == 1:
                target_data.append(true_value_list[i])
        ten_value_list = seperate_to_10(target_data)
        ten_value_list /= len(target_data)
        target_data_list.append(ten_value_list)
    plot_hist("Histalgram of true value in CI (clt, t, chi, f)", len(method_list), target_data_list, 0, 10, 0, 0.25)

# get percentage success graph of 10 intervals
def percentage_success(filename):
    true_value_list,target_vals_list,clt_inrange,t_inrange,chi_inrange,f_inrange = loadF(filename)
    method_list = [clt_inrange,t_inrange,chi_inrange,f_inrange]
    target_data_list = []
    for m in range(len(method_list)):
        analyze_method = method_list[m]
        target_data = []
        opposite_data = []
        for i in range(len(analyze_method)):
            if analyze_method[i] == 1:
                target_data.append(true_value_list[i])
            else:
                opposite_data.append(true_value_list[i])
        ten_value_list = seperate_to_10(target_data)
        opposite_ten_value_list = seperate_to_10(opposite_data)
        
        target_data_list.append(ten_value_list/(ten_value_list+opposite_ten_value_list))
    plot_hist("Histalgram of percentage success (clt, t, chi, f)", len(method_list), target_data_list)    

# quick print of some data
def check_data(filename):
    true_value_list,target_vals_list,clt_inrange,t_inrange,chi_inrange,f_inrange = loadF(filename)
    print(np.mean(target_vals_list), len(target_vals_list))

# get coverage for every case
def get_info_coverage(filename, info_type):
    true_value_list,target_vals_list,clt_inrange,t_inrange,chi_inrange,f_inrange = loadF(filename)
    coverage_list_clt = np.zeros(1210)
    coverage_list_t   = np.zeros(1210)
    coverage_list_chi = np.zeros(1210)
    coverage_list_f   = np.zeros(1210)
    case_target_val   = [[] for _ in range(1210)]
    
    for i in range(len(target_vals_list)):
        coverage_list_clt[i%1210] += clt_inrange[i]
        coverage_list_t[i%1210]   += t_inrange[i]
        coverage_list_chi[i%1210] += chi_inrange[i]
        coverage_list_f[i%1210]   += f_inrange[i]
        case_target_val[i%1210].append(target_vals_list[i])
    
    saveF(case_target_val, "mean_"+str(int(len(target_vals_list)/1210))+"r.pkl")
    
    # save variance
    var_mean_list = []
    for i in range(len(case_target_val)):
        var_mean_list.append(np.var(case_target_val[i]))
    saveF(var_mean_list, "var_mean_"+str(int(len(target_vals_list)/1210))+"r.pkl")
    
    coverage_list_clt /= (len(clt_inrange)/1210)
    coverage_list_t   /= (len(clt_inrange)/1210)
    coverage_list_chi /= (len(clt_inrange)/1210)
    coverage_list_f   /= (len(clt_inrange)/1210)
    
    saveF(coverage_list_clt, "coverage_clt.pkl")
    saveF(coverage_list_t, "coverage_t.pkl")
    saveF(coverage_list_chi, "coverage_chi.pkl")
    saveF(coverage_list_f, "coverage_f.pkl")
    
    overall_cov_clt = np.mean(coverage_list_clt)
    overall_cov_t   = np.mean(coverage_list_t)
    overall_cov_chi = np.mean(coverage_list_chi)
    overall_cov_f   = np.mean(coverage_list_f)
    print("overall coverage (clt,t,chi,f):",overall_cov_clt,overall_cov_t,overall_cov_chi,overall_cov_f)
    
    if info_type == "min":
        cut_point_clt = np.percentile(coverage_list_clt, 0.49)
        cut_point_t   = np.percentile(coverage_list_t, 0.49)
        cut_point_chi = np.percentile(coverage_list_chi, 0.49)
        cut_point_f   = np.percentile(coverage_list_f, 0.49)
        
        print("min of clt,t,chi,f ; 0.5 percentile:")
        print(min(coverage_list_clt), cut_point_clt)
        print(min(coverage_list_t), cut_point_t)
        print(min(coverage_list_chi), cut_point_chi)
        print(min(coverage_list_f), cut_point_f)
        
        ind_list_clt = []
        ind_list_t   = []
        ind_list_chi = []
        ind_list_f   = [] 
        
        for i in range(1210):
            if coverage_list_clt[i] <= cut_point_clt:
                ind_list_clt.append(i)
            if coverage_list_t[i] <= cut_point_t:
                ind_list_t.append(i)
            if coverage_list_chi[i] <= cut_point_chi:
                ind_list_chi.append(i)
            if coverage_list_f[i] <= cut_point_f:
                ind_list_f.append(i)              
                
        print("indlist:",ind_list_clt,ind_list_t,ind_list_chi,ind_list_f)
        saveF([ind_list_clt,ind_list_t,ind_list_chi,ind_list_f], "ind_"+str(int(len(target_vals_list)/1210))+"r.pkl")
        print("ind_"+str(int(len(target_vals_list)/1210))+"r.pkl")
        
    else:
        cut_point_clt = np.percentile(coverage_list_clt, 99.5)
        cut_point_t   = np.percentile(coverage_list_t, 99.5)
        cut_point_chi = np.percentile(coverage_list_chi, 99.5)
        cut_point_f   = np.percentile(coverage_list_f, 99.5)
        
        print("max of clt,t,chi,f ; 99.5 percentile:")
        print(max(coverage_list_clt), cut_point_clt)
        print(max(coverage_list_t), cut_point_t)
        print(max(coverage_list_chi), cut_point_chi)
        print(max(coverage_list_f), cut_point_f)
        
        ind_list_clt = []
        ind_list_t   = []
        ind_list_chi = []
        ind_list_f   = [] 
        
        for i in range(1210):
            if coverage_list_clt[i] >= cut_point_clt:
                ind_list_clt.append(i)
            if coverage_list_t[i] >= cut_point_t:
                ind_list_t.append(i)
            if coverage_list_chi[i] >= cut_point_chi:
                ind_list_chi.append(i)
            if coverage_list_f[i] >= cut_point_f:
                ind_list_f.append(i)              
                
        print("indlist:",ind_list_clt,ind_list_t,ind_list_chi,ind_list_f)
        saveF([ind_list_clt,ind_list_t,ind_list_chi,ind_list_f], "ind_"+str(int(len(target_vals_list)/1210))+"r_max.pkl")
        print("ind_"+str(int(len(target_vals_list)/1210))+"r_max.pkl")        
    
    return

# get pi_l, pi_t and pastar info from index
def get_name_from_index(list_ind):
    str_list = []
    for i in range(len(list_ind)):
        pi_t = list_ind[i]%11
        astar = (list_ind[i]//11)%10
        pi_l = (list_ind[i]//11//10)%11
        str_list.append(str(list_ind[i])+ ' -- l:'+str(pi_l)+', t:'+str(pi_t)+', pastar:'+str(astar))
    return str_list, pi_l, pi_t, astar

def analyze_mean(mean_file, ind_file, ind = None):
    mean_list = loadF(mean_file)
    ind_list_clt,ind_list_t,ind_list_chi,ind_list_f = loadF(ind_file)
    total_list = ind_list_clt+ind_list_t+ind_list_chi+ind_list_f
    total_list = list(dict.fromkeys(total_list))    
    mean_list = np.array(mean_list).astype(np.float)
    if ind != None:
        if ind == "inf":
            for i in total_list:
                print("=====================================================")
                target_mean_list = mean_list[i]
                # normality
                print(stats.normaltest(target_mean_list))
                min_x = min(target_mean_list)
                max_x = max(target_mean_list)
                print("min and max:", min_x,max_x)
                target_mean_list -= min_x
                target_mean_list /= max_x - min_x
                [ind_name],l,t,p = get_name_from_index([i])
                print(ind_name)
                # get variance
                variance = get_variance_info(l, t, p)
                print("variance:", variance)
                ten_val_list = seperate_to_10(target_mean_list)/len(target_mean_list)
                plot_hist(str(len(mean_list[0]))+"r Histalgram of mean value with small coverage ("+ind_name[:3]+")", 1, [ten_val_list], min_x, max_x)   
        else:
            target_mean_list = mean_list[total_list[ind]]
            print("min and max:", min(target_mean_list),max(target_mean_list))
            target_mean_list -= min(target_mean_list)
            target_mean_list /= max(target_mean_list) - min(target_mean_list)
            [ind_name,l,t,p] = get_name_from_index([total_list[ind]])
            print(ind_name)
            ten_val_list = seperate_to_10(target_mean_list)/len(target_mean_list)
            plot_hist(str(len(mean_list[0]))+"r Histalgram of mean value with small coverage ("+ind_name[:3]+")", 1, [ten_val_list])
        
    else:
        target_mean_list = np.concatenate(mean_list[total_list])
        print(get_name_from_index(total_list))
        ten_val_list = seperate_to_10(target_mean_list)/len(target_mean_list)
        plot_hist(str(len(mean_list[0]))+"r Histalgram of mean value with small coverage", 1, [ten_val_list])

    return

def scatter_plot(coverage_file,seperate_variance = False, variance_file = None):
    coverage = loadF(coverage_file)
    if seperate_variance == False:
        # get x and y
        x = []
        y = coverage
        for l in range(11):
            for p in range(10):
                for t in range(11):
                    variance = get_variance_info(l, t, p)
                    x.append(variance)
        
        # calculate covariance
        covariance = stats.pearsonr(x,y)
        print(covariance)
        
        # plot
        plt.scatter(x,y,s=0.6,color='blue')
        plt.xlabel('variance by formula')
        plt.ylabel('coverage')
        plt.show()
    elif seperate_variance == "mean":
        variance = loadF(variance_file)
        # calculate covariance
        covariance = stats.pearsonr(variance,coverage)
        print(covariance)        
        plt.scatter(variance,coverage,s=0.6,color='blue')
        plt.xlabel('variance by 401 instance')
        plt.ylabel('coverage')
        plt.show()     
    elif seperate_variance == "VvsV":
        variance_ins = loadF(variance_file)
        x = []
        for l in range(11):
            for p in range(10):
                for t in range(11):
                    variance = get_variance_info(l, t, p)
                    x.append(variance)     
        variance_formula = x
        # calculate covariance
        covariance = stats.pearsonr(variance_ins,variance_formula)     
        print(covariance)
        plt.scatter(variance_ins,variance_formula,s=0.6,color='blue')
        plt.xlabel('variance by 401 instance')
        plt.ylabel('variance by formula')
        plt.show()           
    
    else:
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        i = 0
        for l in range(11):
            for p in range(10):
                for t in range(11):
                    variance = get_variance_info(l, t, p)
                    if variance <= seperate_variance:
                        x1.append(variance)     
                        y1.append(coverage[i])
                    else:
                        x2.append(variance)
                        y2.append(coverage[i])
                    i += 1
        print("len for <= "+ str(seperate_variance)+": "+str(len(x1)))
        print("len for >  "+ str(seperate_variance)+": "+str(len(x2)))
        # calculate covariance
        covariance1 = stats.pearsonr(x1,y1)
        covariance2 = stats.pearsonr(x2,y2)
        print("covarance <= "+str(seperate_variance),covariance1)     
        print("covarance >  "+str(seperate_variance),covariance2)  
        # plot
        plt.scatter(x1,y1,s=0.7,color='blue')
        plt.xlabel('variance by formula <= '+str(seperate_variance))
        plt.ylabel('coverage')
        plt.show()      
        plt.scatter(x2,y2,s=0.7,color='blue')
        plt.xlabel('variance by formula > '+str(seperate_variance))
        plt.ylabel('coverage')
        plt.show()         

# main function
def main():
    #analyze_value("100r_CI_result.pkl")
    #analyze_success("400r_CI_result.pkl")
    #percentage_success("400r_CI_result.pkl")
    #check_data("400r_CI_result1.pkl")
    #get_info_coverage("400r_CI_result.pkl","max")  #this
    #analyze_mean("mean_401r.pkl","ind_401r.pkl")
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",8)
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",7)
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",6)
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",5)
    #analyze_mean("mean_401r.pkl","ind_401r.pkl","inf")
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",3)
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",2)
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",1)
    #analyze_mean("mean_401r.pkl","ind_401r.pkl",0)
    scatter_plot("coverage_t.pkl","VvsV","var_mean_401r.pkl")
    print("DONE!")
    
main()