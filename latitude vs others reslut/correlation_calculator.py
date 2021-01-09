#***********************
#Wang Dong
#Wild Fire Project
#***********************

#@@@@@@@@@@@@@@@@@@@@@
# imported - not builtin
import csv
import pickle
import numpy as np
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
# get data from data list 
def getData(var_name):
    var_list = readcsv("data.csv")[0]
    data = pickle.load(open(fname, "rb"))
    var_index = var_list.index(var_name)
    return_data = data[var_index][1:]  
    return return_data
#-----------------------------------------------------------------
# get listed data from data list 
def getListedData(var_name):
    ori_data = readcsv("data.csv")
    var_list = ori_data[0]
    var_index = var_list.index(var_name)
    return_data = []
    for i in range (len(ori_data)):
        if i > 0:
            return_data.append([ori_data[i][var_index]])
    return return_data
#-----------------------------------------------------------------
# get listed data list from data list 
def getListedDataList(var_list_given):
    ori_data = readcsv("data.csv")
    var_list = ori_data[0]
    return_data = []
    for v_index in range(len(var_list_given)):
        var_index = var_list.index(var_list_given[v_index])
        # ignore catigorical variable
        if not ori_data[1][var_index][0].isalpha():
            if v_index == 0:
                for i in range (len(ori_data)):
                    if i > 0:
                        return_data.append([ori_data[i][var_index]])
            else:
                for i in range (len(ori_data)):
                    if i > 0:
                        return_data[i-1].append(ori_data[i][var_index])            
    return return_data
#-----------------------------------------------------------------
# get correlation from two variable lists
def getCorrCoef(first_data_name, second_data_name):
    first_data = getData(first_data_name)
    second_data = getData(second_data_name)    
    r = np.corrcoef(np.array(first_data).astype(np.float),np.array(second_data).astype(np.float))
    r = "%.3f"% round(r[0][1],3)
    f = open("out.txt", "a")
    min_len = 20
    if len(second_data_name)<min_len:
        data_name = first_data_name + ": " + second_data_name+" "*(min_len-len(second_data_name))  
    else:
        data_name = first_data_name + ": " + second_data_name 
    f.write("Correlation r {0} = {1}\n".format(data_name, r))
    f.close()
    print("correlation saved in out.txt file")
    return r
#=========================================================================


# main function
def main():


    calculate_CC = False
      
    if calculate_CC:
        getCorrCoef('latitude', 'ia_success')
        getCorrCoef('latitude', 'max_size')
        getCorrCoef('latitude', 'first_size')      
        getCorrCoef('latitude', 'first_status_held')
        getCorrCoef('latitude', 'sec_to_uc')    
        getCorrCoef('latitude', 'aircraft_n_total')
        getCorrCoef('latitude', 'aircraft_hr_total')
        getCorrCoef('latitude', 'n_firefighters')
        getCorrCoef('latitude', 'n_non_firefighters')
        getCorrCoef('latitude', 'hr_firefighters')
        getCorrCoef('latitude', 'drop_amount_retardant')
        getCorrCoef('latitude', 'drop_amount_water')
        getCorrCoef('latitude', 'drop_amount_total')
        getCorrCoef('latitude', 'n_fire_past_1')
        getCorrCoef('latitude', 'n_fire_past_7')
        getCorrCoef('latitude', 'n_fire_past_30')
        getCorrCoef('latitude', 'response_time')
        getCorrCoef('latitude', 'general_cause')
        getCorrCoef('latitude', 'assessment_size')
        getCorrCoef('latitude', 'fire_spread_rate')
        getCorrCoef('latitude', 'temperature')
        getCorrCoef('latitude', 'relative_humidity')
        getCorrCoef('latitude', 'wind_speed')
        getCorrCoef('latitude', 'equipment_Transportation')
        getCorrCoef('latitude', 'equipment_Water_Delivery')
        getCorrCoef('latitude', 'equipment_Fire_Guard_Building')    
        getCorrCoef('latitude', 'equipment_Base_Camp')
        getCorrCoef('latitude', 'wstation_dry_bulb_temperature')
        getCorrCoef('latitude', 'wstation_relative_humidity')
        getCorrCoef('latitude', 'wstation_wind_speed_kmh')
        getCorrCoef('latitude', 'wstation_precipitation')
        getCorrCoef('latitude', 'wstation_fine_fuel_moisture_code')
        getCorrCoef('latitude', 'wstation_duff_moisture_code')
        getCorrCoef('latitude', 'wstation_drought_code')
        getCorrCoef('latitude', 'wstation_build_up_index')
        getCorrCoef('latitude', 'wstation_initial_spread_index')
        getCorrCoef('latitude', 'wstation_fire_weather_index')
        getCorrCoef('latitude', 'wstation_daily_severity_rating')
        getCorrCoef('latitude', 'fuelgrid_C')
        getCorrCoef('latitude', 'fuelgrid_D')    
        getCorrCoef('latitude', 'fuelgrid_M')
        getCorrCoef('latitude', 'fuelgrid_Nonfuel')
        getCorrCoef('latitude', 'fuelgrid_O')    
        getCorrCoef('latitude', 'fuelgrid_S')
        getCorrCoef('latitude', 'fuelgrid_Unclassified')
        getCorrCoef('latitude', 'fuelgrid_Water')
        getCorrCoef('latitude', 'fbp_CFB')    
        getCorrCoef('latitude', 'fbp_CFC')
        getCorrCoef('latitude', 'fbp_HFI')
        getCorrCoef('latitude', 'fbp_RAZ')
        getCorrCoef('latitude', 'fbp_ROS')    
        getCorrCoef('latitude', 'fbp_SFC')
        getCorrCoef('latitude', 'fbp_TFC')
        getCorrCoef('latitude', 'fbp_HFI_class')    
        getCorrCoef('latitude', 'test_i')     
        print("done!")
    
main()