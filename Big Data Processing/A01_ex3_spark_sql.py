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
import pyspark.sql.functions
from pyspark.sql.functions import *
import time
import datetime
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql.window import Window
# ------------------------------------------
# FUNCTION my_main
# ------------------------------------------

def sortbytime(data):
    day1,time1 = data.split(" ")
    y1,m1,d1 = day1.split("-")
    h1,mi1,s1 = time1.split(":")
    dt1 = datetime(int(y1), int(m1), int(d1), int(h1), int(mi1), int(s1))
    secs = dt1.timestamp()
    return str(secs)

def filterData(data, atstop, current_time, current_stop, seconds_horizon):
    day,time = current_time.split(" ")
    y,m,d = day.split("-")
    h,mi,s = time.split(":")
    day1,time1 = data.split(" ")
    y1,m1,d1 = day1.split("-")
    h1,mi1,s1 = time1.split(":")

    dt = datetime(int(y), int(m), int(d), int(h), int(mi), int(s))
    dt1 = datetime(int(y1), int(m1), int(d1), int(h1), int(mi1), int(s1))
    # dt1 = datetime(2018, 10, 52, 36, 0)
    delta = dt1-dt
    seconds = delta.total_seconds()
    if (seconds < 0) or (seconds > seconds_horizon):
        return False
    elif (atstop == 0):
        return False
    return True

def my_main(spark, my_dataset_dir, current_time, current_stop, seconds_horizon):
    # 1. We define the Schema of our DF.
    my_schema = pyspark.sql.types.StructType(
        [pyspark.sql.types.StructField("date", pyspark.sql.types.StringType(), False),
         pyspark.sql.types.StructField("busLineID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("busLinePatternID", pyspark.sql.types.StringType(), False),
         pyspark.sql.types.StructField("congestion", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("longitude", pyspark.sql.types.FloatType(), False),
         pyspark.sql.types.StructField("latitude", pyspark.sql.types.FloatType(), False),
         pyspark.sql.types.StructField("delay", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("vehicleID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("closerStopID", pyspark.sql.types.IntegerType(), False),
         pyspark.sql.types.StructField("atStop", pyspark.sql.types.IntegerType(), False)
         ])

    # 2. Operation C2: 'read' to create the DataFrame from the dataset and the schema
    inputDF = spark.read.format("csv") \
        .option("delimiter", ",") \
        .option("quote", "") \
        .option("header", "false") \
        .schema(my_schema) \
        .load(my_dataset_dir)

    # ---------------------------------------
    # TO BE COMPLETED
    # ---------------------------------------
#     daysplit_udf = udf(lambda x: x.split(" ")[0].split("-")[2], pyspark.sql.types.StringType())

    filterudf = udf(lambda x,y: filterData(x,y,current_time,current_stop,seconds_horizon), pyspark.sql.types.BooleanType())
    filterDF = inputDF.filter(filterudf(col("date"), col("atStop")))
    
    sortimeUDF = udf(lambda x: sortbytime(x), pyspark.sql.types.StringType())
    sec_columnDF = filterDF.withColumn("secs", sortimeUDF(col("date")))
    onlygoodcolDF = sec_columnDF.select(sec_columnDF["date"], sec_columnDF["vehicleID"], sec_columnDF["closerStopID"])
    onlygoodcolDF.persist()
        
    df1 = onlygoodcolDF.filter(col("closerstopID") == current_stop)
    
    df2 = df1.first()
    
    vehicletofilter = df2["vehicleID"]
    
    df3 = onlygoodcolDF.filter(col("vehicleID") == vehicletofilter).withColumnRenamed("closerstopID", "stop").withColumnRenamed("date", "time")
    
    new_df = df3.select(
      'vehicleID',
      struct(col("time"), col("stop")).alias('stations')
    )
    
    solutionDF = new_df.groupBy(col("vehicleID")).agg(collect_list(col("stations"))).withColumnRenamed("collect_list(stations)", "stations")
    

    # ---------------------------------------

#     Operation A1: 'collect' to get all results
    resVAL = solutionDF.collect()
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

    # 4. We configure the Spark Session
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    print("\n\n\n")

    # 5. We call to our main function
    my_main(spark, my_dataset_dir, current_time, current_stop, seconds_horizon)