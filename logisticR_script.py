from __future__ import division  # floating point division
import csv
import random
import math
import pickle

import correlation_calculator as cc
import data_processing as dp
import numpy as np
import utilities as utils
import classalgorithms as algs
import matplotlib.pyplot as plt


def classify():
    # init variables
    run = True
    plot = True
    trainsize = 12500
    testsize = 12500
    numruns = 1
    k_fold = False
    dataset_file = "data.csv"

    classalgs = {
        'Logistic Regression': algs.LogitReg()
    }
    numalgs = len(classalgs)

    num_steps = 1
    parameters = (
        {'regularizer': 'None', 'stepsize':0.001, 'num_steps':num_steps, 'batch_size':2},
        #{'regularizer': 'None', 'stepsize':0.01, 'num_steps':300, 'batch_size':20},
                      )
    numparams = len(parameters)

    accuracy = {}
    for learnername in classalgs:
        accuracy[learnername] = np.zeros((numparams,numruns))

    # load dataset & shuffle 
    dataset = dp.readcsv(dataset_file)
    Y = cc.getData("ia_success")
    Y = np.array(Y).astype(np.float)
    #X = cc.getListedData("fbp_HFI")
    X = cc.getListedDataList(['fbp_CFB', 'fbp_CFC', 'fbp_HFI', 'fbp_RAZ', 'fbp_ROS', 'fbp_SFC', 'fbp_TFC', 'fbp_HFI_class'])
    #X = cc.getListedDataList(['assessment_result',  'max_size', 'first_size', 'first_status_held', 'sec_to_uc', 'aircraft_n_Fixed', 'aircraft_n_Rotary', 'aircraft_n_total', 'aircraft_hr_Fixed', 'aircraft_hr_Rotary', 'aircraft_hr_total', 'n_firefighters', 'n_non_firefighters', 'hr_firefighters', 'hr_non_firefighters', 'drop_amount_retardant', 'drop_amount_water', 'drop_amount_total', 'n_fire_past_1',
                             #'n_fire_past_7', 'n_fire_past_30', 'response_time', 'general_cause', 'year', 'month', 'latitude', 'longitude', 'assessment_size', 'fire_spread_rate', 'fire_position_on_slope', 'temperature', 'relative_humidity', 'wind_direction', 'wind_speed', 'weather_conditions_over_fire', 'equipment_Transportation', 'equipment_Water_Delivery', 'equipment_Sustained_Action', 'equipment_Fire_Guard_Building',
                             #'equipment_Crew_Gear', 'equipment_Base_Camp', 'equipment_WaterTruck_Transportation', 'wstation_dry_bulb_temperature', 'wstation_relative_humidity', 'wstation_wind_speed_kmh', 'wstation_wind_direction', 'wstation_precipitation', 'wstation_fine_fuel_moisture_code', 'wstation_duff_moisture_code', 'wstation_drought_code', 'wstation_build_up_index', 'wstation_initial_spread_index', 'wstation_fire_weather_index', 'wstation_daily_severity_rating', 'fuelgrid_C', 'fuelgrid_D', 'fuelgrid_M', 'fuelgrid_Nonfuel', 'fuelgrid_O',
                             #'fuelgrid_S', 'fuelgrid_Unclassified', 'fuelgrid_Water', 'fuel_type2', 'grouped_fuel_type2', 'fbp_CFB', 'fbp_CFC', 'fbp_FD', 'fbp_HFI', 'fbp_RAZ', 'fbp_ROS', 'fbp_SFC', 'fbp_TFC', 'fbp_HFI_class', 'fuel_type', 'grouped_fuel_type', 'test_i'
                             #])
    X = cc.getListedDataList(['max_size', 'first_size', 'first_status_held', 'sec_to_uc', 'aircraft_n_Fixed', 'aircraft_n_Rotary', 'aircraft_n_total', 'aircraft_hr_Fixed', 'aircraft_hr_Rotary', 'aircraft_hr_total', 'n_firefighters', 'n_non_firefighters', 'hr_firefighters', 'hr_non_firefighters', 'drop_amount_retardant', 'drop_amount_water', 'drop_amount_total', 'n_fire_past_1',
                             'n_fire_past_7', 'n_fire_past_30', 'response_time', 'general_cause', 'year', 'month', 'latitude', 'longitude', 'assessment_size', 'fire_spread_rate', 'fire_position_on_slope', 'temperature', 'relative_humidity', 'wind_direction', 'wind_speed', 'weather_conditions_over_fire', 'equipment_Transportation', 'equipment_Water_Delivery', 'equipment_Sustained_Action', 'equipment_Fire_Guard_Building',
                             'equipment_Crew_Gear', 'equipment_Base_Camp', 'equipment_WaterTruck_Transportation', 'wstation_dry_bulb_temperature', 'wstation_relative_humidity', 'wstation_wind_speed_kmh', 'wstation_wind_direction', 'wstation_precipitation', 'wstation_fine_fuel_moisture_code', 'wstation_duff_moisture_code', 'wstation_drought_code', 'wstation_build_up_index', 'wstation_initial_spread_index', 'wstation_fire_weather_index', 'wstation_daily_severity_rating', 'fuelgrid_C', 'fuelgrid_D', 'fuelgrid_M', 'fuelgrid_Nonfuel', 'fuelgrid_O',
                             'fuelgrid_S', 'fuelgrid_Unclassified', 'fuelgrid_Water', 'fuel_type2', 'grouped_fuel_type2', 'fbp_CFB', 'fbp_CFC', 'fbp_FD', 'fbp_HFI', 'fbp_RAZ', 'fbp_ROS', 'fbp_SFC', 'fbp_TFC', 'fbp_HFI_class', 'fuel_type', 'grouped_fuel_type', 'test_i'
                             ])    
    #print(X)
    X = np.array(X).astype(np.float)
    #trainX, testX = pickle. load(open(dataset_file, "rb"))
    
    #trainY = np.append(np.zeros(len(trainX[0][2500:])),np.ones(len(trainX[1][2500:])))
    #testY = np.append(np.zeros(len(testX[0])),np.ones(len(testX[1])))
    #valY = np.append(np.zeros(2500),np.ones(2500))
    
    #valX = np.append(trainX[0][:2500], trainX[1][:2500], axis=0)
    #trainX = np.append(trainX[0][2500:], trainX[1][2500:], axis=0)
    #testX = np.append(testX[0], testX[1], axis=0)  
    
    np.random.seed(3111)
    np.random.shuffle(X)
    np.random.seed(3111)
    np.random.shuffle(Y)
    
    trainX = X[:len(X)//2]
    valX = X[len(X)//2: len(X)*3//4]
    testX = X[len(X)*3//4:]
    
    trainY = Y[:len(Y)//2]
    valY = Y[len(Y)//2: len(Y)*3//4]
    testY = Y[len(Y)*3//4:]    
    
    
    # Run
    if run:
        for r in range(numruns):
            print(('Running on train={0}, val={1}, test={2} samples for run {3}').
                  format(trainX.shape[0], valX.shape[0], testX.shape[0], r))
    
            # test different parameters (only one for this assignment)
            for p in range(numparams):
                params = parameters[p]
                
                # only one algorithm for now
                for learnername, learner in classalgs.items():
                    # Reset learner for new parameters
                    learner.reset(params)
                    print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                    # Train model
                    #print("trainset0: ", trainset[0])
                    learner.learn(trainX, trainY, valX, valY, testX, testY)
                    # Test model
                    predictions = learner.predict(testX)
                    acc = utils.getaccuracy(testY, predictions)
                    print ('accuracy for ' + learnername + ': ' + str(acc))
                    accuracy[learnername][p,r] = acc
                
    # plot
    if plot == True:
        print("PLOT!")
        accuracy_val, accuracy_test, accuracy_train, best_accuracy, best_weight = pickle. load(  open("learning_acc.pkl", "rb")) 
        print("best_accuracy : val,train,test", accuracy_val, accuracy_train, accuracy_test)
        epi = np.arange(0, num_steps, 1)
        plt.plot(epi,accuracy_val, label='validation accuracy : 1')
        plt.plot(epi,accuracy_test, label='test accuracy : 2')
        plt.plot(epi,accuracy_train, label='train accuracy : 3')
        plt.xlabel('epochs')
        plt.ylabel('Accuracy %') 
        plt.legend()    
        plt.show()


def main():
    classify()
    
    print("--------------done!-----------------")
    
    
main()