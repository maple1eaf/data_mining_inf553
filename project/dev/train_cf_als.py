"""
vocareum
python3 train.py
local
spark-submit train.py
"""

import sys
import time
import json
from math import sqrt
from projectsupport import *

from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

import platform

system_type = platform.system()
if system_type == 'Linux':
    print(system_type)
    # for run on vocareum
    import os
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

    train_file = "../resource/asnlib/publicdata/train_review.json"
    model_file = "model.json"
elif system_type == 'Darwin':
    print(system_type)
    # run for local macos
    train_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/train_review.json"
    itemcf_model_file = '/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/model_itemCF.json'
    als_model_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/als.json"
    checkpoint_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/dev/checkpoint"
else:
    print('wrong system type.')
    sys.exit()

CORATED_LIMIT = 10
LONELY_USER_THRESHOLD = 5
LONELY_BUSINESS_THRESHOLD = 5

# # for tuning
# CORATED_LIMIT = int(sys.argv[1])
# LONELY_USER_THRESHOLD = int(sys.argv[2])
# LONELY_BUSINESS_THRESHOLD = int(sys.argv[3])
# itemcf_model_file = '/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/model_itemCF.json'
# als_model_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/als.json"

def pc(b1, b2):
    # b1, b2 - (bid, {uid: star, ...})
    # avg_business - {bid: avg, ...}
    b1_bid = b1[0]
    b2_bid = b2[0]
    b1_d = b1[1]
    b2_d = b2[1]
    b1_u = set(b1_d.keys())
    b2_u = set(b2_d.keys())
    u_intersect = list(b1_u.intersection(b2_u))
    len_inter = len(u_intersect)
    if len_inter < CORATED_LIMIT:
        return None
    b1_corated = [b1_d[uid] for uid in u_intersect]
    b2_corated = [b2_d[uid] for uid in u_intersect]

    # co-rated average
    b1_avg = meanList(b1_corated)
    b2_avg = meanList(b2_corated)

    b1_corated_normalized = [(x - b1_avg) for x in b1_corated]
    b2_corated_normalized = [(x - b2_avg) for x in b2_corated]
    n = sum([(b1_corated_normalized[i] * b2_corated_normalized[i]) for i in range(len(u_intersect))])
    d1 = sum([(x * x) for x in b1_corated_normalized])
    d2 = sum([(x * x) for x in b2_corated_normalized])
    if n == 0 or d1 == 0 or d2 == 0:
        return None
    w = n / sqrt(d1 * d2)
    if w > 0:
#         w = w ** 2.5 # Case Amplification
        return (b1_bid, b2_bid, w) # b1_bid < b2_bid
    else:
        return None

def getData(sc):
    raw_data = sc.textFile(train_file) \
        .map(json.loads) \
        .persist(StorageLevel.MEMORY_AND_DISK)
    return raw_data

# for itembased model
def transformDataForItemBased(raw_data):
    data_groupby_bid = raw_data.map(lambda r: ((r['business_id'], r['user_id']), [r['stars']])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: averageRating(x)) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], convertToDict(x[1]))) \
        .collect()
        # .filter(lambda x: len(x[1]) > 2) \
        # .map(lambda x: (x[0], convertToDict(x[1]))) \
        # .collect()

    # print(data_groupby_bid[:2])
    print(len(data_groupby_bid)) # 10253 -> 10118 (remove businesses that were rated by fewer than 2 users)

    return data_groupby_bid

def computePearsonCorrelationItemBased(data):
    # data - [(bid, {uid: star, ...}), ...]
    # avg_business - {bid: avg, ...}
    data_dict = {} # {bid: (bid, {uid: star, ...}), ...}
    for bid, d in data:
        data_dict[bid] = (bid, d)
    bid_list = list(data_dict.keys())
    bid_list.sort()
    bid_length = len(bid_list)
    # bid_pairs = ((bid_list[i], bid_list[j]) for i in range(bid_length) for j in range(i+1, bid_length))
    bid_v = []
    for i in range(bid_length):
        x = bid_list[i]
#         print(i)
        for j in range(i+1, bid_length):
            y = bid_list[j]
            res = pc(data_dict[x], data_dict[y])
            if res != None:
                d_ = {'b1': res[0], 'b2': res[1], 'sim': res[2]}
                # print(d_)
                bid_v.append(d_)
    # print(bid_v[:5])
    print(len(bid_v)) # 214796
    return bid_v

def outputCFModelToFile(model, model_file):
    with open(model_file, 'w', encoding='utf-8') as fp:
        for item in model:
            fp.write(json.dumps(item))
            fp.write('\n')

def collectAlsModel(modelRDD, u_table, b_table):
    user_featrue = modelRDD.userFeatures() \
        .map(lambda x: (u_table[x[0]], list(x[1])[0])) \
        .collectAsMap()
    product_feature = modelRDD.productFeatures() \
        .map(lambda x: (b_table[x[0]], list(x[1])[0])) \
        .collectAsMap()
    return [user_featrue, product_feature]

def saveAlsModel(modelRDD, u_table, b_table, model_file):
    model = collectAlsModel(modelRDD, u_table, b_table)
    with open(model_file, 'w', encoding='utf-8') as fp:
        json.dump(model, fp)

def itembaseCF(sc, raw_data):
    # transform to generate a dataset for item-based model
    data_groupby_bid = transformDataForItemBased(raw_data)

    # compute Pearson Correlation w
    model = computePearsonCorrelationItemBased(data_groupby_bid)

    # output model to file
    outputCFModelToFile(model, itemcf_model_file)

def als(sc, raw_data):
    lonely_user = raw_data.map(lambda x: (x['user_id'], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] < LONELY_USER_THRESHOLD) \
        .map(lambda x: x[0]) \
        .collect()
    lonely_business = raw_data.map(lambda x: (x['business_id'], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] < LONELY_BUSINESS_THRESHOLD) \
        .map(lambda x: x[0]) \
        .collect()

    u_table = raw_data.map(lambda x: x['user_id']) \
        .distinct() \
        .collect()
    b_table = raw_data.map(lambda x: x['business_id']) \
        .distinct() \
        .collect()

    stars_data = raw_data.map(lambda x: ((x['user_id'], x['business_id']), [x['stars']])) \
        .filter(lambda x: x[0][0] not in lonely_user and x[0][1] not in lonely_business) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], meanList(x[1]))) \
        .map(lambda x: (u_table.index(x[0][0]), b_table.index(x[0][1]), x[1])) \
        .map(lambda x: Rating(x[0], x[1], x[2])) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    sc.setCheckpointDir(checkpoint_file)
    ALS.checkpointInterval = 2

    modelRDD = ALS.train(ratings=stars_data, rank=1, iterations=70, lambda_=0.01, nonnegative=True)
    saveAlsModel(modelRDD, u_table, b_table, als_model_file)

def train():
    conf = SparkConf() \
        .setAppName("project") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)

    # get data
    raw_data = getData(sc)

    als(sc, raw_data)
    itembaseCF(sc, raw_data)

if __name__ == "__main__":
    t_1 = time.time()
    train()
    print('Time %fs.' % (time.time() - t_1))

