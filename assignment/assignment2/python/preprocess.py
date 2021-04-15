"""
spark-submit preprocess.py <business_file_path> <review_file_path> <output_file_path>

spark-submit preprocess.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/business.json" "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/review.json" "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/task2_data.csv"
"""
import csv
import json
import sys
import time

from pyspark import SparkConf, SparkContext

business_file = sys.argv[1]
review_file = sys.argv[2]
output_file = sys.argv[3]

time0 = time.time()
# define spark env
conf = SparkConf() \
    .setAppName("task1") \
    .setMaster("local[*]")
sc = SparkContext(conf=conf)

business_data = sc.textFile(business_file) \
    .map(json.loads) \
    .map(lambda x: (x["business_id"], x["state"])) \
    .filter(lambda x: x[1] == "NV")

review_data = sc.textFile(review_file) \
    .map(json.loads) \
    .map(lambda x: (x["business_id"], x["user_id"]))

user_business = review_data.join(business_data) \
    .map(lambda x: (x[1][0], x[0])) \
    .collect()

header = ["user_id", "business_id"]

with open(output_file, 'w') as fp:
    wr = csv.writer(fp)
    wr.writerow(header)
    wr.writerows(user_business)
    
time1 = time.time()
print("consume %fs." % (time1-time0))
