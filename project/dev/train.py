"""
vocareum
python3 train.py
local
spark-submit train.py
"""

import json
import os
import platform
import re
import sys
import time

import numpy as np
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

import support

system_type = platform.system()
if system_type == 'Linux':
    print(system_type)
    # for run on vocareum
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

    train_file = "../resource/asnlib/publicdata/train_review.json"
    model_file = "model.json"
elif system_type == 'Darwin':
    print(system_type)
    # run for local macos
    stopwords_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/dev/stopwords"

    train_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/train_review.json"
    user_avg_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/user_avg.json"
    business_avg_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/business_avg.json"
    user_json = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/user.json"
    business_json = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/business.json"

    model_dir = './model/'
    als_model_file = model_dir + "als.json"
    checkpoint_file = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/project/dev/checkpoint"
    agm_train_file = model_dir + "agm_train.json"
else:
    print('wrong system type.')
    sys.exit()

JS_THRESHOLD = 0.7
AGM_USER_THRESHOLD = 8
AGM_THRESHOLD = 3
UNK = 3.7961611526341503
LONELY_USER_THRESHOLD = 5
LONELY_BUSINESS_THRESHOLD = 8

# # for tuning
# CORATED_LIMIT = int(sys.argv[1])
# LONELY_USER_THRESHOLD = int(sys.argv[2])
# LONELY_BUSINESS_THRESHOLD = int(sys.argv[3])
# itemcf_model_file = '/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/model_itemCF.json'
# als_model_file = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/model/als.json"

def processCategories(v, stopwords):
    # v - "Arcades, Arts & Entertainment"
    v = v.lower()
    pattern = r"[a-z]+" # only words
    words_without_punc_num = re.findall(pattern, v)
    words_clean = set([word for word in words_without_punc_num if word not in stopwords])
    return words_clean

def computeJaccardSimilarity(i_set, j_set):
    fenzi = len(i_set.intersection(j_set))
    fenmu = len(i_set.union(j_set))
    return fenzi / fenmu

def getJS(i_b, b_profile, b_list):
    i_set = b_profile[i_b]
    l_ = []
    for j in range(len(b_list)):
        j_b = b_list[j]
        if j_b > i_b:
            j_set = b_profile[j_b]
            sim = computeJaccardSimilarity(i_set, j_set)
            if sim >= JS_THRESHOLD and sim != 0.0:
                new_1 = (i_b, [(j_b, sim)])
                new_2 = (j_b, [(i_b, sim)])
                l_.append(new_1)
                l_.append(new_2)
    return l_

def adjustedSim(sim, target, accord, n_b_avg):
    t_avg = n_b_avg.get(target, UNK)
    a_avg = n_b_avg.get(accord, UNK)
    if a_avg > t_avg:
        return sim
    else:
        return 1 / sim

def processValues(vs, jaccard_sim, n_b_avg):
    # vs - [(n_b, star), ...]
    # jaccard_sim - [(0, {1:0.7, ...}), ...]
    if len(vs) >= AGM_USER_THRESHOLD or len(vs) < AGM_THRESHOLD:
        return vs
    v_d = {k: v for k, v in vs}
    v_d_keys = set(v_d.keys())
    vs_agm = []
    for x in jaccard_sim:
        target_b = x[0]
        if target_b not in v_d_keys:
            sim_b = x[1]
            sim_b_keys = set(sim_b.keys())
            inter = list(v_d_keys.intersection(sim_b_keys))
            if len(inter) >= AGM_THRESHOLD and len(inter) != 0:
                v_vct = np.array([v_d[k] for k in inter])
                b_vct_fenzi = np.array([adjustedSim(sim_b[k], target_b, k, n_b_avg) for k in inter])
                b_vct = np.array([sim_b[k] for k in inter])
                
                agm_stars = np.dot(v_vct, b_vct_fenzi) / b_vct.sum()
                if agm_stars > 5.0:
                    agm_stars = 5.0
                vs_agm.append((target_b, agm_stars))
    return vs + vs_agm

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

def train():
    conf = SparkConf() \
        .setAppName("project") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)

    # check model dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # rename
    raw_data = sc.textFile(train_file).map(json.loads).persist(StorageLevel.MEMORY_AND_DISK)
    u_table1 = raw_data.map(lambda x: x['user_id']).distinct().collect()
    u_set1 = set(u_table1)
    b_table1 = raw_data.map(lambda x: x['business_id']).distinct().collect()
    b_set1 = set(b_table1)

    user_avg = support.getAvg(user_avg_file)
    business_avg = support.getAvg(business_avg_file)
    u_set2 = set(user_avg.keys())
    b_set2 = set(business_avg.keys())

    b_table3 = sc.textFile(business_json).map(json.loads).map(lambda x: x['business_id']).collect()
    b_set3 = set(b_table3)

    u_table = list(u_set1.union(u_set2))
    b_table = list(b_set1.union(b_set2).union(b_set3))
    u_d = {u_table[i]: i for i in range(len(u_table))}
    b_d = {b_table[i]: i for i in range(len(b_table))}

    # agmentation
    business_avg = support.getAvg(business_avg_file)
    n_b_avg = {b_d[k]: business_avg[k] for k in business_avg}

    # get stopwords
    stopwords = sc.textFile(stopwords_file).collect()

    b_profile = sc.textFile(business_json) \
        .map(json.loads) \
        .map(lambda x: (x['business_id'], x['categories'])) \
        .map(lambda x: (b_d[x[0]], x[1])) \
        .mapValues(lambda v: processCategories(v, stopwords)) \
        .collectAsMap()
    b_list = list(sorted(b_profile.keys()))
    b_length = len(b_profile)
    jaccard_sim = sc.parallelize(b_list) \
        .flatMap(lambda x: getJS(x, b_profile, b_list)) \
        .reduceByKey(lambda x, y: x + y) \
        .mapValues(lambda vs: {k: v for k, v in vs}) \
        .collect()

    agm_data = raw_data.map(lambda r: (r['user_id'], r['business_id'], r['stars'])) \
        .map(lambda x: (u_d[x[0]], b_d[x[1]], x[2])) \
        .map(lambda x: (x[0], [(x[1], x[2])])) \
        .reduceByKey(lambda x, y: x + y) \
        .mapValues(lambda vs: processValues(vs, jaccard_sim, n_b_avg)) \
        .flatMap(lambda x: [(x[0], b, star) for b, star in x[1]]) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    # asl
    agm_train = agm_data.map(lambda x: ((u_table[x[0]], b_table[x[1]]), x[2])).collect()
    support.writeDownRenameTable(agm_train, agm_train_file)

    lonely_user = agm_data.map(lambda x: (x[0], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] < LONELY_USER_THRESHOLD) \
        .map(lambda x: x[0]) \
        .collect()
    lonely_business = agm_data.map(lambda x: (x[1], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] < LONELY_BUSINESS_THRESHOLD) \
        .map(lambda x: x[0]) \
        .collect()

    stars_data = agm_data.filter(lambda x: x[0] not in lonely_user and x[1] not in lonely_business) \
        .map(lambda x: Rating(x[0], x[1], x[2])).persist(StorageLevel.MEMORY_AND_DISK)
    sc.setCheckpointDir(checkpoint_file)
    ALS.checkpointInterval = 2
    modelRDD = ALS.train(ratings=stars_data, rank=1, iterations=70, lambda_=0.01, nonnegative=True)
    saveAlsModel(modelRDD, u_table, b_table, als_model_file)

if __name__ == "__main__":
    t_1 = time.time()
    train()
    print('Time %fs.' % (time.time() - t_1))
