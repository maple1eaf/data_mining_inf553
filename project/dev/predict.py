"""
python3 predict.py <test_file> <output_file>
python3 predict.py "../resource/asnlib/publicdata/test_review.json" "prediction.json"
spark-submit predict.py <test_file> <output_file>
"""

import sys
import time
import json

from pyspark import SparkConf, SparkContext, StorageLevel
import support

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
    test_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/test_review.json"
    output_file = "../predict/prediction_als.json"

    train_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/train_review.json"
    als_model_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/als.json"

    agm_train_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/agm_train.json"
    business_avg_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/business_avg.json"
else:
    print('wrong system type.')
    sys.exit()

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

def loadAlsModel(model_file):
    with open(model_file, 'r', encoding='utf-8') as fp:
        model = json.load(fp)
    user_feature = model[0]
    product_feature = model[1]
    als_model = support.Als()
    als_model.setModel(user_feature, product_feature)
    return als_model

def outputResultToFile(prediction, output_file):
    with open(output_file, 'w', encoding='utf-8') as fp:
        for item in prediction:
            t = {
                'user_id': item[0][0],
                'business_id': item[0][1],
                'stars': item[1]
            }
            fp.write(json.dumps(t))
            fp.write('\n')

def dealwithNone(v, bid, business_avg):
    if v != None:
        return v
    b_avg = business_avg.get(bid, business_avg['UNK'])
    return b_avg

def normalX(x):
    if x == None:
        return x
    if x > 5.0:
        x = 5.0
    elif x < 0.0:
        x = 0.0
    else:
        x = x
    return x

def predictAlsOrTrain(uid, bid, als_model, train_data):
    res1 = train_data.get((uid, bid))
    res2 = als_model.predict(uid, bid)
    res = None
    if res1 != None and res2 != None:
        res = res1 * 0.3 + res2 * 0.7
    elif res1 == None and res2 != None:
        res = res2
    elif res1 != None and res2 == None:
        res = res1
    else:
        res = None
    res = normalX(res2)
    return res

def predict():
    conf = SparkConf() \
        .setAppName("project") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    business_avg = support.getAvg(business_avg_file)

    agm_train_l = support.readRenameTable(agm_train_file)
    train_data = sc.parallelize(agm_train_l).map(lambda x: (tuple(x[0]), [x[1]])) \
        .reduceByKey(lambda x, y: x + y) \
        .mapValues(lambda vs: support.meanList(vs)) \
        .collectAsMap()
    
    # without none
    als_model = loadAlsModel(als_model_file)
    prediction = sc.textFile(test_file) \
        .map(json.loads) \
        .map(lambda x: (x['user_id'], x['business_id'])) \
        .map(lambda x: (x, predictAlsOrTrain(x[0], x[1], als_model, train_data))) \
        .map(lambda x: (x[0], dealwithNone(x[1], x[0][1], business_avg))) \
        .collect()
    outputResultToFile(prediction, output_file)

if __name__ == "__main__":
    t_start = time.time()
    predict()
    t_end = time.time()
    print('time: %fs' % (t_end-t_start))