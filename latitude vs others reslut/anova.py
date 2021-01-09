import pandas as pd
import data_processing as d_p
import pickle
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

fname = "processed_data.pkl"

data = pickle.load(open(fname, "rb"))
latitude = d_p.getData('latitude')
assessment_r = d_p.getData('assessment_result')

ar_type = []
group_data = []
for i in range(len(assessment_r)):
    if assessment_r[i] not in ar_type:
        ar_type.append(assessment_r[i])
        group_data.append([])
        
    var_index = ar_type.index(assessment_r[i])
    group_data[var_index].append(latitude[i])
print(ar_type)

fv, pv = stats.f_oneway(group_data[0],group_data[1],group_data[2],group_data[3],group_data[4])
print(fv,pv)
    
temp_data = []
for i in range(len(ar_type)):
    temp_data.append( np.array(group_data[i]).astype(np.float))
group_data = temp_data

fig, ax = plt.subplots()
ax.set_title('Multiple assessment result with latitude')
ax.boxplot(group_data)
ax.set_xlabel('1=Immediate Action, 2=Delayed Action - No Resources Available, 3=Modified Action, 4=Delayed Action - Lower Priority, 5=Beyond Resources Capability')
ax.set_ylabel('latitude')

plt.show()