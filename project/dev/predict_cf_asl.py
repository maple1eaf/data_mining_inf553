"""
python3 predict.py <test_file> <output_file>
python3 predict.py "../resource/asnlib/publicdata/test_review.json" "prediction.json"
spark-submit predict.py <test_file> <output_file>
"""

import sys
import time
import json

from pyspark import SparkConf, SparkContext, StorageLevel
from projectsupport import *

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
elif system_type == 'Darwin':
    print(system_type)
    # run for local macos
    train_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/train_review.json"
    test_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/test_review.json"
    # als_model_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/als.json"
    # itemcf_model_file = 'file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/model_itemCF.json'
    als_model_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/als_5_5.json"
    itemcf_model_file = 'file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/model_itemCF.json'
    user_avg_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/user_avg.json"
    business_avg_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/business_avg.json"
    output_file = "../predict/prediction_combine.json"
else:
    print('wrong system type.')
    sys.exit()

N_NEIGHBORS_ITEMBASED = 10
DEFAULT_OUTPUT = None
WEIGHT = 0.3

# # for tuning
# N_NEIGHBORS_ITEMBASED = int(sys.argv[1])
# DEFAULT_OUTPUT = None
# WEIGHT = float(sys.argv[2])

def predictICF(corated, target_bid, model):
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
    
    if len(collect) < N_NEIGHBORS_ITEMBASED:
        return None
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
#         predict_stars = n / sum_w
#         origin_n = Decimal(str(predict_stars))
#         ans_n = origin_n.quantize(Decimal('0'), rounding=ROUND_HALF_UP)
#         return float(ans_n)





def getData(sc):
    train_raw_data = sc.textFile(train_file) \
        .map(json.loads)
    
    test_raw_data = sc.textFile(test_file) \
        .map(json.loads) \
        .map(lambda x: (x['user_id'], x['business_id'])) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    return train_raw_data, test_raw_data

def getModelItembased(sc, model_file):
    model = sc.textFile(model_file) \
        .map(json.loads) \
        .map(lambda r: ((r['b1'], r['b2']), r['sim'])) \
        .collectAsMap()
    # model - {(bid1, bid2): sim, ...}  ps: bid1 < bid2
    return model

def transformTrainDataGroupByUid(train_raw_data):
    u_d = train_raw_data.map(lambda r: ((r['user_id'], r['business_id']), [r['stars']])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: averageRating(x)) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], convertToDict(x[1]))) \
        .collectAsMap()
    # u_d - {uid: {bid: star, ...}, ...}
    return u_d

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

def loadAlsModel(model_file):
    with open(model_file, 'r', encoding='utf-8') as fp:
        model = json.load(fp)
    user_feature = model[0]
    product_feature = model[1]
    als_model = Als()
    als_model.setModel(user_feature, product_feature)
    return als_model

def fitRange(x):
    if x > 5.0:
        x = 5.0
    elif x < 0.0:
        x = 0.0
    else:
        x = x
    return x

def decidePrediction(p_cf, p_als, weight):
    res = None
    if p_cf != None and p_als != None:
        res = p_cf * weight + p_als * (1 - weight)
        # res = p_cf
        res =  fitRange(res)
    elif p_cf == None and p_als != None:
        res = p_als
        res =  fitRange(res)
    elif p_cf != None and p_als == None:
        res = p_cf
        res =  fitRange(res)
    else:
        res = None
    return res

def getMean(v1, v2):
    return (v1 + v2) / 2

def dealwithNone(v, uid, bid, user_avg, business_avg):
    if v != None:
        return v
    u_avg = user_avg.get(uid, user_avg['UNK'])
    b_avg = business_avg.get(bid, business_avg['UNK'])
    _avg = getMean(u_avg, b_avg)
    return b_avg

def predict():
    conf = SparkConf() \
        .setAppName("project") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    user_avg = getAvg(user_avg_file)
    business_avg = getAvg(business_avg_file)

    # get raw data
    train_raw_data, test_raw_data = getData(sc)
    # get itembased cf model
    itemcf_model = getModelItembased(sc, itemcf_model_file)
    # transform to generate a dataset for item-based model
    u_d = transformTrainDataGroupByUid(train_raw_data)

    # get als model
    als_model = loadAlsModel(als_model_file)

    # predict by itembased cf
    prediction = test_raw_data.map(lambda x: (x, (predictICF(u_d.get(x[0]), x[1], itemcf_model), als_model.predict(x[0], x[1])))) \
        .map(lambda x: (x[0], decidePrediction(x[1][0], x[1][1], WEIGHT))) \
        .map(lambda x: (x[0], dealwithNone(x[1], x[0][0], x[0][1], user_avg, business_avg))) \
        .collect()

    # output prediction to file
    outputResultToFileItembased(prediction)


if __name__ == "__main__":
    t_start = time.time()
    predict()
    t_end = time.time()
    print('time: %fs' % (t_end-t_start))