from statistics import *
import pandas as pd
import data_processing as d_p
import pickle
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

fname = "processed_data.pkl"

data = pickle.load(open(fname, "rb"))
latitude = d_p.getData('latitude')
latitude = np.array(latitude).astype(np.float)

cat = True
plot_cat = True
num = False
plot_num = False


# seperate latitude to 3 lists [L,M,H]
latitude_list = [[],[],[]]
L_M = 53.9047
M_H = 59.0688
for i in range(len(latitude)):
    if latitude[i] < L_M:
        latitude_list[0].append(i)
    elif latitude[i] >= M_H:
        latitude_list[2].append(i)
    else:
        latitude_list[1].append(i)
#print(len(latitude_list[0]),len(latitude_list[1]),len(latitude_list[2])) #(1374 7547 768)
    

data = pickle.load(open(fname, "rb"))
characteristic = [[],[],[]]
labels = []
cat_L = []
cat_M = []
cat_H = []
mean_L = []
mean_M = []
mean_H = []
sd_L = []
sd_M = []
sd_H = []
median_L = []
median_M = []
median_H = []
categories = []
for i in range(len(data)):
    if str(data[i][1])[0].isalpha() or str(data[i][0])=='month' or str(data[i][0])=='general_cause':
        if cat and str(data[i][0])!='wind_direction' and str(data[i][0])!='fuel_type2' and str(data[i][0])!='grouped_fuel_type2' and str(data[i][0])!='grouped_fuel_type':
            print(data[i][0])
            categories.append(data[i][0])
            npdata = np.array(data[i][1:]).astype(np.str)
            L = npdata[latitude_list[0]]
            M = npdata[latitude_list[1]]
            H = npdata[latitude_list[2]]
            
            type = []
            group_data = [[],[],[]]
            for j in range(len(L)):
                if L[j] not in type:
                    type.append(L[j])
                    group_data[0].append(0)   
                    group_data[1].append(0)
                    group_data[2].append(0)
                var_index = type.index(L[j])
                group_data[0][var_index] += 1
            for j in range(len(M)):
                if M[j] not in type:
                    type.append(M[j])
                    group_data[0].append(0)   
                    group_data[1].append(0)
                    group_data[2].append(0)
                var_index = type.index(M[j])
                group_data[1][var_index] += 1        
            for j in range(len(H)):
                if H[j] not in type:
                    type.append(H[j])
                    group_data[0].append(0)   
                    group_data[1].append(0)
                    group_data[2].append(0)
                var_index = type.index(H[j])
                group_data[2][var_index] += 1 
            labels = labels + type
            print(type)
            #print(group_data)
            # get % for each category
            for q in range(len(group_data)):
                s = sum(group_data[q])
                for p in range(len(group_data[q])):
                    group_data[q][p] = group_data[q][p]/s
            print(group_data)
            cat_L += group_data[0]
            cat_M += group_data[1]
            cat_H += group_data[2]
            #print(group_data)
    elif num:
        npdata = np.array(data[i][1:]).astype(np.float)
        L = npdata[latitude_list[0]]
        M = npdata[latitude_list[1]]
        H = npdata[latitude_list[2]]   
        labels.append(data[i][0])
        mean_L.append(mean(L))
        mean_M.append(mean(M))
        mean_H.append(mean(H))
        sd_L.append(stdev(L))
        sd_M.append(stdev(M))
        sd_H.append(stdev(H))
        median_L.append(median(L))
        median_M.append(median(M))
        median_H.append(median(M))
        

 
if cat:      
    print(labels)
    print(categories)
    labels[0] = 'Im_A'
    labels[1] = 'D_A_LP'
    labels[2] = 'M_A'
    labels[3] = 'BRC'
    labels[4] = 'D_A_NRA'
    labels[5] = 'Other'
    labels[6] = 'Lightning'

if plot_cat:
    length = 14
    x = np.arange(14)
    fig, ax = plt.subplots()
    width = 0.35
    rects1 = ax.bar(x , cat_H[30:], width)
    #rects2 = ax.bar(x - width/6, cat_M[(length*7)//2:], width/3, label='M')
    #rects3 = ax.bar(x + width/6, cat_H[(length*7)//2:], width/3, label='H')
    ax.set_ylabel('% of category')
    ax.set_title('summary of fuel_type(H)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels[30:])
    ax.legend()
    ax.set_ylim(0,1)
    fig.tight_layout()
    plt.show()
#print(cat_L)
#print(cat_M)
#print(cat_H)

if num:
    with open("3_regions_num.txt","w") as f:
        f.write("%-25s " %" ")
        f.write("%13s " %"Mean_Low")
        f.write("%13s " %"Mean_Mid")
        f.write("%13s " %"Mean_High")
        f.write("%13s " %"Stdv_Low")
        f.write("%13s " %"Stdv_Mid")
        f.write("%13s " %"Stdv_Hi")
        f.write("%13s " %"Median_L")
        f.write("%13s " %"Median_M")
        f.write("%13s " %"Median_H")
        f.write("\n")
        for i in range(len(labels)):
            f.write("%-25s " %labels[i])
            f.write("%13.4f " %round(mean_L[i],4))
            f.write("%13.4f " %round(mean_M[i],4))
            f.write("%13.4f " %round(mean_H[i],4))
            f.write("%13.4f " %round(sd_L[i],4))
            f.write("%13.4f " %round(sd_M[i],4))
            f.write("%13.4f " %round(sd_H[i],4))
            f.write("%13.4f " %round(median_L[i],4))
            f.write("%13.4f " %round(median_M[i],4))
            f.write("%13.4f " %round(median_H[i],4))
            f.write("\n")
        
        #for i in mean_L:
            
        #f.write("\n")
        
        #for i in mean_M:
            #f.write("%-10s " %i)
        #f.write("\n")
        
        #for i in mean_H:
            #f.write("%-10s " %i)
        #f.write("\n")
        
        #for i in sd_L:
            #f.write("%-10s " %i)
        #f.write("\n")
        
        #for i in sd_M:
            #f.write("%-10s " %i)
        #f.write("\n")
        
        #for i in sd_H:
            #f.write("%-10s " %i)
        #f.write("\n")
        
        #for i in median_L:
            #f.write("%-10s " %i)
        #f.write("\n")
        
        #for i in median_M:
            #f.write("%10s " %i)
        #f.write("\n")
        
        #for i in median_H:
            #f.write("%10s " %i)
        #f.write("\n")

if plot_num:
    length = len(labels)
    x = np.arange(length)
    fig, ax = plt.subplots()
    width = 0.35
    rects1 = ax.bar(x - width/2, mean_L, width/3, label='L')
    rects2 = ax.bar(x - width/6, mean_M, width/3, label='M')
    rects3 = ax.bar(x + width/6, mean_H, width/3, label='H')
    ax.set_ylabel('% of category')
    ax.set_title('summary of categorical variables')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()    