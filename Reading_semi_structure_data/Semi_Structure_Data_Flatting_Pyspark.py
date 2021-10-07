#!/usr/bin/env python
# coding: utf-8

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
import json

spark = SparkSession         .builder         .appName("Python Spark SQL Basic Example")         .getOrCreate()

from pyspark.sql.functions import col, explode_outer
from pyspark.sql.types import *
from copy import deepcopy
from collections import Counter
import pyspark.sql.functions as f

#method to flat the json file 
def flatten(df):  
    complex_fields = dict([(field.name, field.dataType)
                             for field in df.schema.fields
                             if type(field.dataType) == ArrayType or  type(field.dataType) == StructType])
    while len(complex_fields)!=0:
        col_name=list(complex_fields.keys())[0]
        print ("Processing Column:"+col_name+" Type of column : "+str(type(complex_fields[col_name])))
    
        if (type(complex_fields[col_name]) == StructType):
            expanded = [col(col_name+'.'+k).alias(col_name+'_'+k) for k in [ n.name for n in  complex_fields[col_name]]]
            df=df.select("*", *expanded).drop(col_name)
    
        elif (type(complex_fields[col_name]) == ArrayType):    
            df=df.withColumn(col_name,explode_outer(col_name))
        
        complex_fields = dict([(field.name, field.dataType)
                             for field in df.schema.fields
                             if type(field.dataType) == ArrayType or  type(field.dataType) == StructType])
        print(len(complex_fields))
    return df
json_df1 = spark.read           .option("multiline",'true')           .json('test_json.json')
json_schema = json_df1.schema
json_df1.printSchema()


json_df1.schema.names

#if needed to skip any column you can use this here i have skipped updateDescription as it's defined in two places
names = json_df1.schema.names
df2 = []
for col_name in names:
    if ((isinstance(json_df1.schema[col_name].dataType, StructType)) and col_name == 'updateDescription'):
        print("Skipping struct column %s "%(col_name))
    else:
        print(col_name)
        df2.append(col_name)

json_df1 = json_df1.select(df2)

df2 = flatten(json_df1)

df2.count()
df2.show()
