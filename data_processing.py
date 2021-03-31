#***********************
#Wang Dong
#Wild Fire Project
#***********************

#@@@@@@@@@@@@@@@@@@@@@
# imported - not builtin
import csv
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
#@@@@@@@@@@@@@@@@@@@@@

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# GLOBAL VARIABLE
fname = "processed_data.pkl"
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#=========================================================================
# FUNCTION DEFINATION

#-----------------------------------------------------------------
# read csv and append data into process_data1 variable
def readcsv(file_name):
    process_data1 = []
    with open(file_name, newline='') as csvdata:
        reader = csv.reader(csvdata, delimiter=',', quotechar='|')
        for row in reader:
            process_data1.append(row)
    return process_data1
#-----------------------------------------------------------------
# reshape data into vertical
def reshape(process_data1):
    num_v_col_1 = len(process_data1[0])
    num_d_ver_2 = len(process_data1)       
    process_data_vertical = []
    for i in range(num_v_col_1):
        process_data_vertical.append([])
    
    for row in process_data1:
        for col_num in range(num_v_col_1):
            process_data_vertical[col_num].append(row[col_num])
    return process_data_vertical
#-----------------------------------------------------------------
# change "TRUE", "FALSE" to 1, 0
def changeTF(process_data_vertical):
    num_v_col_1 = len(process_data_vertical)
    num_d_ver_2 = len(process_data_vertical[0])       
    num_TRUE = 0
    num_FALSE = 0
    for row in range(num_v_col_1):
        for col in range(num_d_ver_2):
            if process_data_vertical[row][col] == "TRUE":
                num_TRUE+=1
                process_data_vertical[row][col] = 1
            elif process_data_vertical[row][col] == "FALSE":
                num_FALSE+=1
                process_data_vertical[row][col] = 0

    #print(num_TRUE)
    #print(num_FALSE)
    return process_data_vertical
#-----------------------------------------------------------------
# change "lightning" and "other" to 1, 0 (in general_cause)
def changeLO(process_data_vertical):
    num_v_col_1 = len(process_data_vertical)
    num_d_ver_2 = len(process_data_vertical[0])         
    num_lighting = 0
    num_other = 0
    for row in range(num_v_col_1):
        for col in range(num_d_ver_2):
            if process_data_vertical[row][col] == "lightning":
                num_lighting+=1
                process_data_vertical[row][col] = 1
            elif process_data_vertical[row][col] == "other":
                num_other+=1
                process_data_vertical[row][col] = 0
    
    #print(num_lighting)
    #print(num_other)
    return process_data_vertical
#-----------------------------------------------------------------
# save data into .pkl file, fname = "abc.pkl"
def saveF(data,fname):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, fname)
    pickle.dump(data, open(fname, "wb"))
#-----------------------------------------------------------------
# load .pkl file
def loadF(fname): 
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, fname)    
    data = pickle.load(open(fname, "rb"))
    return data
#-----------------------------------------------------------------
# get data from data list 
def getData(var_name):
    var_list = readcsv("data.csv")[0]
    data = pickle.load(open(fname, "rb"))
    var_index = var_list.index(var_name)
    return_data = data[var_index][1:]  
    return return_data
#-----------------------------------------------------------------
# get pi_l and pi_t from Mostafa
def get_pil_pit():
    k = 10
    file_name = "df_sim_k10_h1.csv"
    file_list = readcsv(file_name)[1:]
    pi_l_list = []
    pi_t_list = []
    for l in range(k+1):
        pi_l_list.append([])
        pi_t_list.append([])
        for i in range(k):
            pi_l_list[l].append(file_list[l*110+l*10+i][3])
            pi_t_list[l].append(file_list[l*110+l*10+i][4])
    pi_l_list = np.array(pi_l_list).astype(np.float)
    pi_t_list = np.array(pi_t_list).astype(np.float)
    return pi_l_list, pi_t_list
#-----------------------------------------------------------------
# plot histalgram
def plot_hist(suptitle, l, data, min_x = 0, max_x=10, y_low_limit = 0, y_high_limit = 1, x_label = "mean", y_label = "percentage"):
    fig = plt.figure()
    fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
    #fig.suptitle(suptitle)
    
    width = 0.09*(max_x-min_x)
    x = np.arange(len(data[0]))
    
    x = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])*(max_x-min_x)+min_x
    print(x)
    
    for i in range (0,l):
        ax = fig.add_subplot(1, l, i+1)
        ax.bar(x, data[i], width)
        
        plt.ylim(y_low_limit,y_high_limit)
    
    for ax in fig.get_axes():
        ax.label_outer()
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
    fig.savefig(suptitle+'.pdf',bbox_inches='tight')
    return
#-----------------------------------------------------------------
# lower triangle matrix
def get_T(k=10):
    T = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i>=j:
                T[i][j] = 1    
    return T
#=========================================================================

'''
All the variable names
['assessment_result', 'ia_success', 'max_size', 'first_size', 'first_status_held', 'sec_to_uc', 'aircraft_n_Fixed', 'aircraft_n_Rotary', 'aircraft_n_total', 'aircraft_hr_Fixed', 'aircraft_hr_Rotary', 'aircraft_hr_total', 'n_firefighters', 'n_non_firefighters', 'hr_firefighters', 'hr_non_firefighters', 'drop_amount_retardant', 'drop_amount_water', 'drop_amount_total', 'n_fire_past_1']
['n_fire_past_7', 'n_fire_past_30', 'response_time', 'general_cause', 'year', 'month', 'latitude', 'longitude', 'assessment_size', 'fire_spread_rate', 'fire_position_on_slope', 'temperature', 'relative_humidity', 'wind_direction', 'wind_speed', 'weather_conditions_over_fire', 'equipment_Transportation', 'equipment_Water_Delivery', 'equipment_Sustained_Action', 'equipment_Fire_Guard_Building']
['equipment_Crew_Gear', 'equipment_Base_Camp', 'equipment_WaterTruck_Transportation', 'wstation_dry_bulb_temperature', 'wstation_relative_humidity', 'wstation_wind_speed_kmh', 'wstation_wind_direction', 'wstation_precipitation', 'wstation_fine_fuel_moisture_code', 'wstation_duff_moisture_code', 'wstation_drought_code', 'wstation_build_up_index', 'wstation_initial_spread_index', 'wstation_fire_weather_index', 'wstation_daily_severity_rating', 'fuelgrid_C', 'fuelgrid_D', 'fuelgrid_M', 'fuelgrid_Nonfuel', 'fuelgrid_O']
['fuelgrid_S', 'fuelgrid_Unclassified', 'fuelgrid_Water', 'fuel_type2', 'grouped_fuel_type2', 'fbp_CFB', 'fbp_CFC', 'fbp_FD', 'fbp_HFI', 'fbp_RAZ', 'fbp_ROS', 'fbp_SFC', 'fbp_TFC', 'fbp_HFI_class', 'fuel_type', 'grouped_fuel_type', 'test_i']
'''

# main function
def main():
    
    process_data = False
    if process_data:
        process_data1 = readcsv("data.csv")
        process_data_vertical = reshape(process_data1)
        process_data_vertical = changeTF(process_data_vertical)
        process_data_vertical = changeLO(process_data_vertical)
        saveF(process_data_vertical, fname)
        
        print("The file is saved as processed_data.pkl in the same folder")
        print("Done!")
    else:
        print("Turn process_data to True to be able to run the data_processing main function (debuging, ignore this)")
    
main()
