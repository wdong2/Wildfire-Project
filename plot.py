import matplotlib.pyplot as plt
import numpy as np
import math
import data_processing as dp


#=======================================================================================
# Function definition
def get_data(filename, index):
    # if the data is [data1, data2] and index is 0, we will only get data1
    data = dp.loadF(filename)
    if index != None:
        data = data[index]
    return data

def get_subtitle(filename, index = None, inverse = False, y_low_limit = None, y_high_limit = None, IS = False):
    # get subtitle based on the parameters
    suptitle_target = "None target defined"
    if filename[:5] == "F_0_1"[:5]:
        if index == 0:
            suptitle_target = "F(a,0)"
        if index == 1:
            suptitle_target = "F(a,1)"
    elif filename[:12] == "worst_p_star_list.pkl"[:12]:
        if IS == False:
            suptitle_target = "Worst-case distribution of the critical value for the f method"    
        else:
            suptitle_target = "Worst-case distribution of the critical value for the IS method"    
    else:
        suptitle_target = filename.split(".")[0]
    suptitle_target += ", "
    suptitle_dimension = "as default"
    if inverse == False:
        suptitle_dimension = "column = login 0 to 10, row = target 0 to 10"
    else:
        suptitle_dimension = "column = target 0 to 10, row = login 0 to 10"
    if y_low_limit == None and y_high_limit == None:
        return suptitle_target+suptitle_dimension
    return suptitle_target+suptitle_dimension + " range(" + str(y_low_limit) + ", " + str(y_high_limit) + ")"

def get_percentage_stats(value, abs_value, total_diff_stats, total_num_stats, total_absdiff_stats):
    if value >= 0.1:
        total_diff_stats[0] += value
        total_num_stats[0] += 1
        total_absdiff_stats[0] += abs_value
        if value >= 0.25:
            total_diff_stats[1] += value
            total_num_stats[1] += 1         
            total_absdiff_stats[1] += abs_value
            if value >= 0.5:
                total_diff_stats[2] += value
                total_num_stats[2] += 1 
                total_absdiff_stats[2] += abs_value
                if value >= 0.75:
                    total_diff_stats[3] += value
                    total_num_stats[3] += 1 
                    total_absdiff_stats[3] += abs_value
    if value <= -0.1:
        total_diff_stats[4] += value
        total_num_stats[4] += 1   
        total_absdiff_stats[4] += abs_value
        if value <= -0.25:
            total_diff_stats[5] += value
            total_num_stats[5] += 1            
            total_absdiff_stats[5] += abs_value
            if value <= -0.5:
                total_diff_stats[6] += value
                total_num_stats[6] += 1 
                total_absdiff_stats[6] += abs_value
                if value <= -0.75:
                    total_diff_stats[7] += value
                    total_num_stats[7] += 1    
                    total_absdiff_stats[7] += abs_value
    return total_diff_stats, total_num_stats, total_absdiff_stats
    
def print_stats(total_diff_stats, total_num_stats, total_data):
    print("rho >= 10%:", total_num_stats[0]/total_data, "Mean Diff:", total_diff_stats[0]/total_num_stats[0])
    print("rho >= 25%:", total_num_stats[1]/total_data, "Mean Diff:", total_diff_stats[1]/total_num_stats[1])
    print("rho >= 50%:", total_num_stats[2]/total_data, "Mean Diff:", total_diff_stats[2]/total_num_stats[2])
    print("rho >= 75%:", total_num_stats[3]/total_data, "Mean Diff:", total_diff_stats[3]/total_num_stats[3])
    print("rho <= -10%:", total_num_stats[4]/total_data, "Mean Diff:", total_diff_stats[4]/total_num_stats[4])
    print("rho <= -25%:", total_num_stats[5]/total_data, "Mean Diff:", total_diff_stats[5]/total_num_stats[5])
    print("rho <= -50%:", total_num_stats[6]/total_data, "Mean Diff:", total_diff_stats[6]/total_num_stats[6])
    print("rho <= -75%:", total_num_stats[7]/total_data, "Mean Diff:", total_diff_stats[7]/total_num_stats[7])
    
def print_summarize(case, num_positive, abs, rho, numdata = 25):
    persent_pos = 0.0
    persent_neg = 0.0
    mean_diff = 0.0
    mean_rho = 0.0
    if case == "all":
        numdata = 121

    persent_pos = num_positive/numdata
    persent_neg = 1-persent_pos
    mean_diff = abs/numdata
    mean_rho = rho/numdata      
        
    print(case, ": positive", persent_pos, "negative", persent_neg, "mean_diff", mean_diff, "mean_rho", mean_rho)


def plot_bar(filename, k, l, IS = True, inverse = False, index = None):
    # draw multiple bar plot for one figure (2d, square shape, data for each plot = k, length = l)
    
    # set parameters for plot
    data = get_data(filename, index)
    x = np.arange(k)
    width = 0.9
    # the limits are capped by the nearest int
    y_low_limit = math.floor(np.min(data)+0.01)
    #y_low_limit = 0
    y_high_limit = math.ceil(np.max(data))
    print("range:",np.min(data),np.max(data))
    suptitle = get_subtitle(filename, index, inverse, y_low_limit, y_high_limit, IS)
    
    # plot
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
    fig.suptitle(suptitle)
    
    for i in range (1,l*l+1):
        ax = fig.add_subplot(l, l, i)
        if inverse:
            ax.bar(x, data[(i-1)//l][(i-1)%l], width)
        else:
            ax.bar(x, data[(i-1)%l][(i-1)//l], width)
        plt.ylim(y_low_limit,y_high_limit)
    
    for ax in fig.get_axes():
        ax.label_outer()
    
    plt.show()
    return
    

def plot_scatter(filename1, filename2, k, l, inverse = False, compare = "abs"):
    data1 = get_data(filename1, None)
    data2 = get_data(filename2, None)
    print("==============",filename1, "-", filename2, compare,"==============")
    data = []
    # <-0.1 <-0.25 <-0.5 <-0.75; >0.1 >0.25 >0.5 >0.75
    total_diff_stats = np.zeros(8)
    total_num_stats = np.zeros(8)
    total_absdiff_stats = np.zeros(8)
    # all cases
    total_pos = 0.0
    total_abs = 0
    total_rho = 0.0
    # L1-5, T6-10
    top_right_pos = 0.0
    top_right_abs = 0
    top_right_rho = 0.0
    # L6-10, T1-5
    bottom_left_pos = 0.0
    bottom_left_abs = 0
    bottom_left_rho = 0.0
    # L1-5, T1-5
    top_left_pos = 0.0
    top_left_abs = 0
    top_left_rho = 0.0
    # L6-10, T6-10
    bottom_right_pos = 0.0
    bottom_right_abs = 0
    bottom_right_rho = 0.0
    # diaginal
    diaginal_pos = 0.0
    diaginal_abs = 0
    diaginal_rho = 0.0    
    
    total_data = l*l
    
    print(data1,data2)
    print(np.max(data1),np.max(data2))
    
    if compare == "abs":
        data = np.subtract(data1,data2)
    elif compare == "rho":
        for i in range(l):
            data.append([])
            for j in range(l):
                value = (data1[i][j]-data2[i][j])/max(data1[i][j],data2[i][j])
                abs_value = data1[i][j]-data2[i][j]
                data[i].append(value)
                total_diff_stats, total_num_stats, total_absdiff_stats = get_percentage_stats(value, abs_value, total_diff_stats, total_num_stats, total_absdiff_stats)
                
                if value > 0:
                    total_pos += 1
                total_abs += abs_value
                total_rho += value
                
                if i==j and i>0:
                    if value > 0:
                        diaginal_pos += 1
                    diaginal_abs += abs_value
                    diaginal_rho += value                    
                if i >= 1 and i <= 5 and j >= 6 and j <= 10:
                    if value > 0:
                        top_right_pos += 1
                    top_right_abs += abs_value
                    top_right_rho += value            
                elif i >= 6 and i <= 10 and j >= 1 and j <= 5:
                    if value > 0:
                        bottom_left_pos += 1
                    bottom_left_abs += abs_value
                    bottom_left_rho += value     
                elif i >= 1 and i <= 5 and j >= 1 and j <= 5:
                    if value > 0:
                        top_left_pos += 1
                    top_left_abs += abs_value
                    top_left_rho += value       
                elif i >= 6 and i <= 10 and j >= 6 and j <= 10:
                    if value > 0:
                        bottom_right_pos += 1
                    bottom_right_abs += abs_value
                    bottom_right_rho += value                      
                
        print_stats(total_absdiff_stats, total_num_stats, total_data)
        print_summarize("all", total_pos, total_abs, total_rho)
        print_summarize("top_right", top_right_pos, top_right_abs, top_right_rho)
        print_summarize("bot_left", bottom_left_pos, bottom_left_abs, bottom_left_rho)
        print_summarize("top_left", top_left_pos, top_left_abs, top_left_rho)
        print_summarize("bot_right", bottom_right_pos, bottom_right_abs, bottom_right_rho)
        print_summarize("diaginal", diaginal_pos, diaginal_abs, diaginal_rho, 10)
                
    x = np.arange(l)
    # the limits are capped by the nearest int
    y_low_limit = np.min(data)-0.1
    #y_low_limit = 0
    y_high_limit = math.ceil(np.max(data))    
    print("range:",np.min(data),np.max(data))
    suptitle_s = "compare"
    if compare == "abs":
        suptitle_s = filename1.split(".")[0] + " - " + filename2.split(".")[0]
    elif compare == "rho":
        suptitle_s = filename1.split(".")[0] + " - " + filename2.split(".")[0] + "/max(" + filename1.split(".")[0] + " , " + filename2.split(".")[0] + ")"
    suptitle = get_subtitle(suptitle_s, None, inverse, np.min(data), np.max(data))
    
    # plot
    if plot:
        fig = plt.figure()
        fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
        fig.suptitle(suptitle)
        if inverse:
            data = np.array(data).astype(np.float)
            data = data.T
        for i in range (1,l+1):
            ax = fig.add_subplot(l//4+1, 4, i)
            ax.plot(x, data[i-1], 'o')
            plt.ylim(y_low_limit,y_high_limit)
        
        for ax in fig.get_axes():
            ax.label_outer()
        
        plt.show()
    
    return
#=======================================================================================

plot = False


def main():
    '''
    filenames: F_0_1_optVar.pkl, F_0_1_optReg.pkl, F_IS.pkl, worst_p_star_list_f_optVar.pkl, worst_p_star_list_f_optReg.pkl, worst_p_star_list_IS.pkl,
        var_f_optVar, var_f_optReg.pkl, var_IS.pkl
    '''
    # plot parameters
    filename = "F_0_1_optVar.pkl"
    filename2 = "var_f_optReg.pkl" # for scatter plot
    index = 1 # 0 or 1 for filename = F_0_1*.pkl and None for others (worst_p_star_list.pkl, F_IS.pkl)
    plot_mode = "bar" # either bar for data or scatter for comparison (compare is "abs" or "rho")
    compare = "rho" # either "rho" for ratio or "abs" for absolute difference
    IS = False # only true for bar plot and filename = F_IS.pkl or worst_p_star_list_IS.pkl
    # without inverse: col = login, row = target
    inverse = False
    k = 10
    l = 11
    if plot_mode == "bar":
        if filename[:5] == "F_0_1_"[:5]:
            plot_bar(filename, k, l, IS, inverse, index)
        else:
            plot_bar(filename, k, l, IS, inverse)
    elif plot_mode == "scatter": # data1 - data2
        plot_scatter(filename, filename2, k, l, inverse, compare)
        plot_scatter("var_IS.pkl", "var_f_optVar.pkl", k, l, inverse, compare)
        plot_scatter("var_f_optReg.pkl", "var_f_optVar.pkl", k, l, inverse, compare)
    
main()