"""
python3 predict.py <test_file> <output_file>
python3 predict.py "../resource/asnlib/publicdata/train_review.json" "prediction.json"
spark-submit predict.py <test_file> <output_file>
"""

import sys
import time
import json
from math import sqrt
from decimal import Decimal, ROUND_HALF_UP

from pyspark import SparkConf, SparkContext, StorageLevel

import platform

system_type = platform.system()
if system_type == 'Linux':
    print(system_type)
    # for run on vocareum
    import os
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

    train_file = "../resource/asnlib/publicdata/train_review.json"
    test_file = sys.argv[1]
    model_file = "model.json"
    output_file = sys.argv[2]
    cf_type = "item_based"
elif system_type == 'Darwin':
    print(system_type)
    # run for local macos
    train_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/train_review.json"
    test_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/test_review.json"
    model_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/model.json"
    output_file = "../predict/prediction_none.json"
    cf_type = "item_based"
else:
    print('wrong system type.')
    sys.exit()

N_NEIGHBORS_ITEMBASED = 5
N_NEIGHBORS_USERBASED = 5
UNK = 3.823989
DEFAULT_OUTPUT = None

def meanList(l):
    return sum(l) / len(l)

def averageRating(x):
    uid = x[0][0]
    bid = x[0][1]
    stars = x[1] # list
    average_stars = meanList(stars)
    return (uid, [(bid, average_stars)])

def convertToDict(l):
    # l - [(bid, star), ...]
    bs = {}
    for bid, star in l:
        bs[bid] = star
    return bs

def computeStarsItembased(corated, target_bid, model):
    """
    corated - {bid: star, ...}
    """
    if corated == None:
        return None
    corated.pop(target_bid, None)
    bid_cor = list(corated.keys())
    collect = []
    for b in bid_cor:
        pair = None
        if b < target_bid:
            pair = (b, target_bid)
        else:
            pair = (target_bid, b)

        # if b == target_bid:
        #     print('same:', pair)
        w = model.get(pair)
        if w != None:
            # pair may not have a value in the model
            # when b == target_bid, pair have no value, too
            collect.append((pair, w, b))
        # else:
        #     collect.append((pair, 0, b))
    # print(collect)
    collect.sort(key=lambda x: x[1], reverse=True)

    neighbors = collect[:N_NEIGHBORS_ITEMBASED]
    sum_w = 0
    n = 0
    for p, w, b in neighbors:
        star = corated[b]
        n += star * w
        sum_w += w
    if sum_w == 0:
        return None
    else:
        return n /sum_w
        # predict_stars = n / sum_w
        # origin_n = Decimal(str(predict_stars))
        # ans_n = origin_n.quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        # return float(ans_n)

# def computeStarsUserBasedOverallAverage(corated, target_uid, model, uid_avg):
#     """
#     corated - {uid: star, ...}
#     """
#     if corated == None:
#         return None
#     corated.pop(target_uid, None)
#     uid_cor = list(corated.keys())
#     collect = []
#     for u in uid_cor:
#         pair = None
#         if u < target_uid:
#             pair = (u, target_uid)
#         else:
#             pair = (target_uid, u)

#         if u == target_uid:
#             print('same:', pair)
#         w = model.get(pair)
#         if w != None:
#             # pair may not have a value in the model
#             # when b == target_bid, pair have no value, too
#             collect.append((pair, w, u))
#         # else:
#         #     collect.append((pair, 0, u))
#     # print(collect)
#     collect.sort(key=lambda x: x[1], reverse=True)

#     neighbors = collect[:N_NEIGHBORS_USERBASED]
#     sum_w = 0
#     n = 0
#     for p, w, u in neighbors:
#         star = corated[u]
#         avg = uid_avg[u]
#         n += (star - avg) * w
#         sum_w += w
#     if sum_w == 0:
#         return None
#     else:
#         return (n / sum_w) + uid_avg[target_uid]

def computeStarsUserBasedCoratedAverage(x, b_d, u_d, model):
    target_bid = x[0]
    target_uid = x[1]
    corated = b_d.get(target_bid) # corated - {uid: star, ...}
    if corated == None:
        return DEFAULT_OUTPUT
    corated.pop(target_uid, None) # if don't have target_uid, return None
    uid_cor = list(corated.keys()) # - [uids]
    w_collect = []
    for u in uid_cor:
        # because pop, u is not able to == target_uid
        pair = None
        if u < target_uid:
            pair = (u, target_uid)
        else:
            pair = (target_uid, u)
        w = model.get(pair)
        if w != None:
            # pair may not have a value in the model
            # when b == target_bid, pair have no value, too
            w_collect.append((pair, w, u))
    w_collect.sort(key=lambda x: x[1], reverse=True)

    neighbors = w_collect[:N_NEIGHBORS_USERBASED]
    # print(neighbors)
    for i in range(len(neighbors)):
        # neighbors[i] - (pair, w, u)
        u_row = u_d[neighbors[i][2]]
        u_row.pop(target_bid, None)
        u_mean = meanList(list(u_row.values()))
        neighbors[i] = (neighbors[i][0], neighbors[i][1], neighbors[i][2], u_mean)
        # neighbors[i] - (pair, w, u, mean)

    target_u_mean = meanList(list(u_d[target_uid].values()))

    sum_w = 0
    n = 0
    for p, w, u, mean in neighbors:
        star = corated[u]
        n += (star - mean) * w
        sum_w += w
    if sum_w == 0:
        return DEFAULT_OUTPUT
    else:
        predict_value = (n / sum_w) + target_u_mean
        if predict_value > 5:
            return 5.0
        elif predict_value < 0:
            return 0.0
        else:
            return predict_value







def getData(sc):
    train_raw_data = sc.textFile(train_file) \
        .map(json.loads)
    
    test_raw_data = sc.textFile(test_file) \
        .map(json.loads)

    return train_raw_data, test_raw_data

def getModelItembased(sc):
    model_list = sc.textFile(model_file) \
        .map(json.loads) \
        .map(lambda r: ((r['b1'], r['b2']), r['sim'])) \
        .collect()
    model = {}
    for pair, sim in model_list:
        model[pair] = sim
    # model - {(bid1, bid2): sim, ...}  ps: bid1 < bid2
    return model

def transformTrainDataGroupByUid(train_raw_data):
    data_groupby_uid = train_raw_data.map(lambda r: ((r['user_id'], r['business_id']), [r['stars']])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: averageRating(x)) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], convertToDict(x[1]))) \
        .collect()
    
    u_d = {}
    for uid, bd in data_groupby_uid:
        u_d[uid] = bd
    # u_d - {uid: {bid: star, ...}, ...}
    return u_d

def predictItembased(model, u_d, test_raw_data):
    print('1')
    prediction = test_raw_data.map(lambda r: (r['user_id'], r['business_id'])).collect()
    print('2')
    for i in range(len(prediction)):
        x = prediction[i]
        prediction[i] = (x, computeStarsItembased(u_d.get(x[0]), x[1], model))
    # prediction = test_raw_data.map(lambda r: (r['user_id'], r['business_id'])) \
    #     .map(lambda x: (x, computeStarsItembased(u_d.get(x[0]), x[1], model))) \
    #     .collect()
    # [((uid, bid), star), ...]
    return prediction

def outputResultToFileItembased(prediction):
    with open(output_file, 'w', encoding='utf-8') as fp:
        for item in prediction:
            t = {
                'user_id': item[0][0],
                'business_id': item[0][1],
                'stars': item[1]
            }
            fp.write(json.dumps(t))
            fp.write('\n')

def getModelUserBased(sc):
    model_list = sc.textFile(model_file) \
        .map(json.loads) \
        .map(lambda r: ((r['u1'], r['u2']), r['sim'])) \
        .collect()
    model = {}
    for pair, sim in model_list:
        model[pair] = sim
    # model - {(uid1, uid2): sim, ...}  ps: uid1 < uid2
    return model

def transformTrainDataGroupByBid(train_raw_data):
    data_groupby_bid = train_raw_data.map(lambda r: ((r['business_id'], r['user_id']), [r['stars']])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: averageRating(x)) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], convertToDict(x[1]))) \
        .collect()
    
    b_d = {}
    for bid, ud in data_groupby_bid:
        b_d[bid] = ud
    # b_d - {bid: {uid: star, ...}, ...}
    return b_d

# for corated average situation
def predictUserBased(model, b_d, u_d, test_raw_data):
    prediction = test_raw_data.map(lambda r: (r['business_id'], r['user_id'])) \
        .map(lambda x: (x, computeStarsUserBasedCoratedAverage(x, b_d, u_d, model))) \
        .collect()
    # [((bid, uid), star), ...]
    return prediction

def outputResultToFileUserBased(prediction):
    with open(output_file, 'w', encoding='utf-8') as fp:
        for item in prediction:
            t = {
                'user_id': item[0][1],
                'business_id': item[0][0],
                'stars': item[1]
            }
            fp.write(json.dumps(t))
            fp.write('\n')

if __name__ == "__main__":
    t_start = time.time()
    # define spark env
    conf = SparkConf() \
        .setAppName("task3predict") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    # get raw data
    train_raw_data, test_raw_data = getData(sc)

    if cf_type == 'item_based':
        # get model
        model = getModelItembased(sc)

        # transform to generate a dataset for item-based model
        u_d = transformTrainDataGroupByUid(train_raw_data)

        # predict star-rating
        prediction = predictItembased(model, u_d, test_raw_data)

        # output prediction to file
        outputResultToFileItembased(prediction)
    elif cf_type == 'user_based':
        # get model
        model = getModelUserBased(sc)

        # transform to generate a dataset for item-based model
        u_d = transformTrainDataGroupByUid(train_raw_data)

        # transform to generate a dataset for user-based model
        b_d = transformTrainDataGroupByBid(train_raw_data)

        # all average from train_review.json
        # avg_all = get

        # predict star-rating
        prediction = predictUserBased(model, b_d, u_d, test_raw_data)

        # output prediction to file
        outputResultToFileUserBased(prediction)
    else:
        print('wrong cf_type value.')
    
    t_end = time.time()
    print('time: %fs' % (t_end-t_start))