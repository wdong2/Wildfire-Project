import csv
import math
import pickle

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import warnings
import classalgorithms as algs
import matplotlib.pyplot as plt


def regression():
    dataset_file = "data.csv"

    df2 = pd.read_csv(dataset_file)
    
    model = sm.GLM.from_formula("ia_success ~  max_size + assessment_result + first_status_held + aircraft_hr_Fixed + n_firefighters + hr_non_firefighters + n_fire_past_1 + n_fire_past_7 + year + fire_spread_rate + relative_humidity + wind_direction + wind_speed + equipment_Fire_Guard_Building + wstation_relative_humidity + fuel_type + latitude", family=sm.families.Binomial(), data=df2)
    
    result = model.fit()
    print(result.summary())     



def main():
    regression()
    
    print("--------------done!-----------------")
    
    
main()