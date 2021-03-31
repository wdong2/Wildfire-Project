#@@@@@@@@@@@@@@@@@@@@@
# imported - not builtin
import csv
import pickle
import numpy as np
#@@@@@@@@@@@@@@@@@@@@@

#######################################
# Read a .csv file into .pkl file
# change filename for .csv file and change filename_save for the .pkl file
# the format of .pkl is a list which has 2 index out.pkl[a][b]: a is column number and b is row number
#######################################


def readP_csv(file_name):
    process_data1 = []
    i = 0
    with open(file_name, newline='') as csvdata:
        reader = csv.reader(csvdata, delimiter=',', quotechar='|')
        for row in reader:
            if i%10 == 0:
                process_data1.append([])
            if i>0:
                process_data1[(i-1)//10].append(row[1])
            i+=1
    return process_data1

# save data into .pkl file, fname = "abc.pkl"
def saveF(data,fname):
    pickle.dump(data, open(fname, "wb"))

def main():
    filename = "fixed-kernel-hdot5.csv"
    filename_save = "hdot5.pkl"
    data = readP_csv(filename)
    saveF(data, filename_save)
    print("Done!")
    
main()