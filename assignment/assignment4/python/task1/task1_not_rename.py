"""
spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py <filter_threshold> <input_file_path> <community_output_file_path>

spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1_dev.py 7 "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw4/data/ub_sample_data.csv" "./output/task1_1_ans.txt"

vocareum:
spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py 7 $ASNLIB/publicdata/ub_sample_data.csv task1_1_ans
"""

import sys
import time
import json

from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import *

# for run on vocareum
# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

LPA_MAXITER = 5

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]

def outputAsFile(res):
    with open(community_output_file_path, 'w', encoding='utf-8') as fp:
        for t in res:
            if len(t) == 1:
                fp.write("'%s'\n" % (t[0]))
            else:
                fp.write("'" + ("', '").join(t) + "'\n")

def task1():
    # define spark env
    conf = SparkConf() \
        .setAppName("task1") \
        .setMaster("local[3]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    data_with_header = sc.textFile(input_file_path)
    header = data_with_header.first()
    baskets = data_with_header.filter(lambda l: l != header) \
        .map(lambda l: tuple(l.split(','))) \
        .distinct() \
        .map(lambda x: (x[0], [x[1]])) \
        .reduceByKey(lambda x, y: x + y) \
        .collect()

    # print(baskets['H6QSYPYJFAW2wGOyn12SYg'])
    uids = [x[0] for x in baskets]
    baskets_length = len(baskets)
    edges_list = []
    vertices_have_link = []
    for i in range(baskets_length):
        uid_i = baskets[i][0]
        bids_i = set(baskets[i][1])
        for j in range(i+1, baskets_length):
            uid_j = baskets[j][0]
            bids_j = set(baskets[j][1])
            if len(bids_i.intersection(bids_j)) >= filter_threshold:
                edges_list.append((uid_i, uid_j, 1))
                edges_list.append((uid_j, uid_i, 1))
                if uid_i not in vertices_have_link:
                    vertices_have_link.append(uid_i)
                if uid_j not in vertices_have_link:
                    vertices_have_link.append(uid_j)
    # print(edges_list[:5])
    # print(len(edges_list)) # 498 * 2 = 996


    # build dataframe context
    spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()
    
    uids_transform = [(uid,) for uid in vertices_have_link]
    vertices = spark.createDataFrame(uids_transform, ["id"])

    edges = spark.createDataFrame(edges_list, ["src", "dst", "relationship"])

    g = GraphFrame(vertices, edges)
    communities = g.labelPropagation(maxIter=LPA_MAXITER)

    result = communities.groupby('label') \
        .agg(F.collect_list('id').alias('collect')) \
        .select('collect') \
        .rdd \
        .map(tuple) \
        .map(lambda x: tuple(sorted(x[0]))) \
        .sortBy(keyfunc=lambda x: x[0]) \
        .sortBy(keyfunc=lambda x: len(x)) \
        .collect()
    
    outputAsFile(result)

if __name__ == "__main__":
    t_start = time.time()
    task1()
    t_end = time.time()
    print('time: %fs' % (t_end-t_start))