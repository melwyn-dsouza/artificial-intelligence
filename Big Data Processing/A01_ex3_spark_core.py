# Databricks notebook source
  # --------------------------------------------------------
#
# PYTHON PROGRAM DEFINITION
#
# The knowledge a computer has of Python can be specified in 3 levels:
# (1) Prelude knowledge --> The computer has it by default.
# (2) Borrowed knowledge --> The computer gets this knowledge from 3rd party libraries defined by others
#                            (but imported by us in this program).
# (3) Generated knowledge --> The computer gets this knowledge from the new functions defined by us in this program.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer first processes this PYTHON PROGRAM DEFINITION section of the file.
# On it, our computer enhances its Python knowledge from levels (2) and (3) with the imports and new functions
# defined in the program. However, it still does not execute anything.
#
# --------------------------------------------------------

import pyspark
import time
import datetime
from datetime import datetime
# ------------------------------------------
# FUNCTION process_line
# ------------------------------------------
def process_line(line):
    # 1. We create the output variable
    res = ()

    # 2. We get the parameter list from the line
    params_list = line.strip().split(",")

    #(00) Date => The date of the measurement. String <%Y-%m-%d %H:%M:%S> (e.g., "2013-01-01 13:00:02").
    #(01) Bus_Line => The bus line. Int (e.g., 120).
    #(02) Bus_Line_Pattern => The pattern of bus stops followed by the bus. String (e.g., "027B1001"). It can be empty (e.g., "").
    #(03) Congestion => On whether the bus is at a traffic jam (No -> 0 and Yes -> 1). Int (e.g., 0).
    #(04) Longitude => Longitude position of the bus. Float (e.g., -6.269634).
    #(05) Latitude = > Latitude position of the bus. Float (e.g., 53.360504).
    #(06) Delay => Delay of the bus in seconds (negative if ahead of schedule). Int (e.g., 90).
    #(07) Vehicle => An identifier for the bus vehicle. Int (e.g., 33304)
    #(08) Closer_Stop => An idenfifier for the closest bus stop given the current bus position. Int (e.g., 7486). It can be no bus stop, in which case it takes value -1 (e.g., -1).
    #(09) At_Stop => On whether the bus is currently at the bus stop (No -> 0 and Yes -> 1). Int (e.g., 0).

    # 3. If the list contains the right amount of parameters
    if (len(params_list) == 10):
        # 3.1. We set the right type for the parameters
        params_list[1] = int(params_list[1])
        params_list[3] = int(params_list[3])
        params_list[4] = float(params_list[4])
        params_list[5] = float(params_list[5])
        params_list[6] = int(params_list[6])
        params_list[7] = int(params_list[7])
        params_list[8] = int(params_list[8])
        params_list[9] = int(params_list[9])

        # 3.2. We assign res
        res = tuple(params_list)

    # 4. We return res
    return res

def filterData(data, current_time, current_stop, seconds_horizon):
    day,time = current_time.split(" ")
    y,m,d = day.split("-")
    h,mi,s = time.split(":")
    day1,time1 = data[0].split(" ")
    y1,m1,d1 = day1.split("-")
    h1,mi1,s1 = time1.split(":")

    dt = datetime(int(y), int(m), int(d), int(h), int(mi), int(s))
    dt1 = datetime(int(y1), int(m1), int(d1), int(h1), int(mi1), int(s1))
    # dt1 = datetime(2018, 10, 52, 36, 0)
    delta = dt1-dt
    seconds = delta.total_seconds()
    if (seconds < 0) or (seconds > seconds_horizon):
        return False
    elif (data[9] == 0):
        return False
    return True

def sortbytime(data):
    day1,time1 = data[0].split(" ")
    y1,m1,d1 = day1.split("-")
    h1,mi1,s1 = time1.split(":")
    dt1 = datetime(int(y1), int(m1), int(d1), int(h1), int(mi1), int(s1))
    secs = dt1.timestamp()
    return secs

def gettuple(data):
    return (data[7], (data[0],data[8]))
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------
def my_main(sc, my_dataset_dir, current_time, current_stop, seconds_horizon):
    # 1. Operation C1: 'textFile' to load the dataset into an RDD
    inputRDD = sc.textFile(my_dataset_dir)

    # ---------------------------------------
    # TO BE COMPLETED
    # ---------------------------------------
    processRDD = inputRDD.map(lambda line: process_line(line)).filter(lambda data: filterData(data,current_time, current_stop, seconds_horizon)).sortBy(lambda data: sortbytime(data))    
    gettupRDD = processRDD.map(lambda data: gettuple(data))
    groupvehicleRDD = gettupRDD.groupByKey().mapValues(lambda values: list(values))
    filterfirststopRDD = groupvehicleRDD.filter(lambda data: data[1][0][1] == current_stop)
    filterfirststopRDD.persist()
    maxlen = filterfirststopRDD.map(lambda data: len(data[1])).max(lambda data: data)
    solutionRDD = filterfirststopRDD.filter(lambda data: len(data[1]) == maxlen)

    # ---------------------------------------
    # Operation A1: 'collect' to get all results
    resVAL = solutionRDD.collect()
    for item in resVAL:
        print(item)

# --------------------------------------------------------
#
# PYTHON PROGRAM EXECUTION
#
# Once our computer has finished processing the PYTHON PROGRAM DEFINITION section its knowledge is set.
# Now its time to apply this knowledge.
#
# When launching in a terminal the command:
# user:~$ python3 this_file.py
# our computer finally processes this PYTHON PROGRAM EXECUTION section, which:
# (i) Specifies the function F to be executed.
# (ii) Define any input parameter such this function F has to be called with.
#
# --------------------------------------------------------
if __name__ == '__main__':
    # 1. We use as many input arguments as needed
    current_time = "2013-01-10 08:59:59"
    current_stop = 1935
    seconds_horizon = 1800

    # 2. Local or Databricks
    local_False_databricks_True = True

    # 3. We set the path to my_dataset and my_result
    my_local_path = "../../../3_Code_Examples/L07-23_Spark_Environment/"
    my_databricks_path = "/"
    my_dataset_dir = "FileStore/tables/6_Assignments/my_dataset_complete/"


    if local_False_databricks_True == False:
        my_dataset_dir = my_local_path + my_dataset_dir
    else:
        my_dataset_dir = my_databricks_path + my_dataset_dir

    # 4. We configure the Spark Context
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(sc, my_dataset_dir, current_time, current_stop, seconds_horizon)