"""
Python: $ spark-submit task3.py <input_file> <output_file> <partition_type> <n_partitions> <n> 
Params:
input_file – the input file (the review dataset) 
output_file – the output file contains your answers 
partition_type – the partition function, either “default” or “customized” 
n_partitions – the number of partitions (only effective for the customized partition function) 
n – the threshold of the number of reviews (see 4.3.1)

default:
spark-submit task3.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "output_task3.json" "default" 50 150 
customized:
spark-submit task3.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "output_task3.json" "customized" 50 150 
"""

import sys
import json
import time
from pyspark import SparkContext, SparkConf

# input_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json"
# output_file = "output_task3.json"
# partition_type = "customized" # either "default" or "customized"
# n_partitions = 30 
# n = 150

input_file = sys.argv[1]
output_file = sys.argv[2]
partition_type = sys.argv[3]
n_partitions = int(sys.argv[4])
n = int(sys.argv[5])

def timer(func):
    """timer decorator"""
    def wrapper(*arg, **karg):
        t1 = time.time()
        res = func(*arg, **karg)
        t2 = time.time()
        delta_t = t2 - t1
        print('time consume:%.2fs' % delta_t)
        return res
    return wrapper

def getNumberOfItems(iter):
    par = list(iter)
    n_par = len(par)
    yield n_par

def write_output_file(result, output_file):
    # write on local disk instead of hdfs
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(result, fp)

def customPartitionFunc(business_id):
    return hash(business_id)

@timer
def main(n_p='default'):
    output = {}

    # define spark env
    conf = SparkConf() \
        .setAppName("task3") \
        .setMaster("local[*]")
    sc = SparkContext(conf=conf)

    review_file_path = input_file
    review_data = None
    if n_p == 'default':
        review_data = sc.textFile(review_file_path) \
            .map(lambda r: [json.loads(r)['business_id'], 1]) \
            .cache()
    else:
        review_data = sc.textFile(review_file_path) \
            .map(lambda r: [json.loads(r)['business_id'], 1]) \
            .partitionBy(n_p, partitionFunc=customPartitionFunc) \
            .cache()

    # get n_partitions
    n_partitions = review_data.getNumPartitions()
    output['n_partitions'] = n_partitions

    # get n_items
    n_items = review_data.mapPartitions(getNumberOfItems).collect()
    output['n_items'] = n_items

    # get result
    businesses_that_have_more_than_n_reviews = review_data \
        .reduceByKey(lambda x, y: x+y) \
        .filter(lambda x: x[1] > n) \
        .map(lambda x: [x[0], x[1]]) \
        .collect()
    output['result'] = businesses_that_have_more_than_n_reviews

    write_output_file(output, output_file)

if __name__ == "__main__":
    if partition_type == "default":
        main()
    elif partition_type == "customized":
        main(n_partitions)
    else:
        print('wrong partition_type value.')

