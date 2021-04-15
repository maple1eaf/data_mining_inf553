"""
spark-submit task3train.py <train_file> <model_file> <cf_type>

<cf_type>: either "item_based" or "user_based"

spark-submit task3train_dev.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw3/data/train_review.json" "./output/task3item.model" item_based
spark-submit task3train_dev.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw3/data/train_review.json" "./output/task3user.model" user_based

vocareum:
python3 task3train.py $ASNLIB/publicdata/train_review.json task3item.model item_based
python3 task3train.py $ASNLIB/publicdata/train_review.json task3user.model user_based
"""

import sys
import time
import json
from math import sqrt
from task3support import *

from pyspark import SparkConf, SparkContext, StorageLevel

# for run on vocareum
# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'


NUM_HASHS = 50
LARGE_NUMBER = sys.maxsize
NUM_BANDS = NUM_HASHS
JACCARD_SIMILARITY_THRESHOLD = 0.01 # >=

# BUSINESS_AVG_FILE = "./../../data/business_avg.json"
# with open(BUSINESS_AVG_FILE, 'r', encoding='utf-8') as fp:
#     bid_avg = json.load(fp)

train_file = sys.argv[1]
model_file = sys.argv[2]
cf_type = sys.argv[3]

def meanList(l):
    return sum(l) / len(l)

def averageRating(x):
    bid = x[0][0]
    uid = x[0][1]
    stars = x[1] # list
    average_stars = meanList(stars)
    return (bid, [(uid, average_stars)])

def convertToDict(l):
    # l - [(uid, star), ...]
    us = {}
    for uid, star in l:
        us[uid] = star
    return us

def pc(b1, b2):
    # b1, b2 - (bid, {uid: star, ...})
    b1_bid = b1[0]
    b2_bid = b2[0]
    # if b1_bid == b2_bid:
    #     return (b1_bid, b2_bid, 1.0) # b1_bid == b2_bid
    b1_d = b1[1]
    b2_d = b2[1]
    b1_u = set(b1_d.keys())
    b2_u = set(b2_d.keys())
    u_intersect = list(b1_u.intersection(b2_u))
    if len(u_intersect) < 3:
        return None
    
    b1_corated = [b1_d[uid] for uid in u_intersect]
    b2_corated = [b2_d[uid] for uid in u_intersect]

    # co-rated average
    b1_avg = meanList(b1_corated)
    b2_avg = meanList(b2_corated)

    # overall average
    # b1_avg = bid_avg[b1_bid]
    # b2_avg = bid_avg[b2_bid]

    # overall average according to train_review.json
    # b1_avg = meanList(list(b1_d.values()))
    # b2_avg = meanList(list(b2_d.values()))

    b1_corated_normalized = [(x - b1_avg) for x in b1_corated]
    b2_corated_normalized = [(x - b2_avg) for x in b2_corated]
    n = sum([(b1_corated_normalized[i] * b2_corated_normalized[i]) for i in range(len(u_intersect))])
    d1 = sum([(x * x) for x in b1_corated_normalized])
    d2 = sum([(x * x) for x in b2_corated_normalized])
    if n == 0 or d1 == 0 or d2 == 0:
        return None
    w = n / sqrt(d1 * d2)
    if w > 0:
        return (b1_bid, b2_bid, w) # b1_bid < b2_bid
    else:
        return None

def updateSignature(s_cur, s_new, length):
    # update s_cur
    for i in range(length):
        if s_new[i] < s_cur[i]:
            s_cur[i] = s_new[i]

def minHash(x, hashs):
    u_id = x[0]
    b_ids = x[1]
    signature = [LARGE_NUMBER] * NUM_HASHS
    for b in b_ids:
        s_ = [hashs[i](b) for i in range(NUM_HASHS)]
        updateSignature(signature, s_, NUM_HASHS)
    res = (u_id, signature)
    return res

def LSH(x, b, r, hash_lsh):
    u_id = x[0]
    signature = x[1]
    ress = []
    for i in range(b):
        v_hash = hash_lsh(signature[i*r:(i+1)*r])
        res = ((v_hash, i), u_id)
        ress.append(res)
    return ress

def generatePairs(l):
    l.sort()
    l_len = len(l)
    res_l = [(l[i],l[j]) for i in range(l_len) for j in range(i+1, l_len)]
    # in each pair, pair[0] < pair[1]
    return res_l

def computeJaccardSimilarity(l1, l2):
    # items in l1 are unique to each other, so does l2
    l1_len = len(l1)
    l2_len = len(l2)
    intersect_set = set(l1).intersection(set(l2))
    inter_len = len(intersect_set)
    union_len = l1_len + l2_len - inter_len
    js = inter_len / union_len
    return js

def getData(sc):
    raw_data = sc.textFile(train_file) \
        .map(json.loads)
    return raw_data

# for itembased model
def transformDataForItemBased(raw_data):
    data_groupby_bid = raw_data.map(lambda r: ((r['business_id'], r['user_id']), [r['stars']])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: averageRating(x)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: len(x[1]) > 2) \
        .map(lambda x: (x[0], convertToDict(x[1]))) \
        .collect()

    # print(data_groupby_bid[:2])
    print(len(data_groupby_bid)) # 10253 -> 10118 (remove businesses that were rated by fewer than 2 users)

    return data_groupby_bid

def computePearsonCorrelationItemBased(data):
    # data - [(bid, {uid: star, ...}), ...]
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

def outputModelToFile(model):
    with open(model_file, 'w', encoding='utf-8') as fp:
        for item in model:
            fp.write(json.dumps(item))
            fp.write('\n')

def getRenameData(raw_data):
    # rename bid
    original_bid = raw_data.map(lambda x: x['business_id']).distinct().collect()
    bid_rename = Rename(original_bid)

    # rename uid
    original_uid = raw_data.map(lambda x: x['user_id']).distinct().collect()
    uid_rename = Rename(original_uid)

    # rename data
    data_renamed = raw_data.map(lambda r: (r['user_id'], r['business_id'], r['stars'])) \
        .map(lambda x: (uid_rename.getNewValue(x[0]), bid_rename.getNewValue(x[1]), x[2])) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    return data_renamed, bid_rename, uid_rename

# for user based model
def getCandidates(data_renamed, bid_rename, uid_rename):
    data_distinct = data_renamed.map(lambda x: (x[0], x[1])) \
        .distinct() \
        .persist(StorageLevel.MEMORY_AND_DISK)
    # data after rename: [(22368, 6616), (22369, 4431), (11224, 2238), (9325, 759), (1829, 4435)]
    # data_distinct - [(uid, bid), ...]
    
    # print('long uid', uid_rename.values_length)
    # print('long bid', bid_rename.values_length)
    # generate hash functions for min-hash
    num_bid = bid_rename.values_length
    print('num_uid =', uid_rename.values_length) # 26,184 can generate 342,787,836 pairs
    print('num_bid =', num_bid) # 10,253 can generate 52,556,878 pairs
    hashs_minhash = generateHashs(NUM_HASHS, num_bid)

    # generate hash functions and parameters for LSH
    b = NUM_BANDS
    r = int(NUM_HASHS / NUM_BANDS)
    hash_lsh = generateHashForLSH(r)

    # generate candidates using Min-Hash & LSH
    candidates = data_distinct.groupByKey() \
        .map(lambda x: minHash(x, hashs_minhash)) \
        .flatMap(lambda x: LSH(x, b, r, hash_lsh)) \
        .groupByKey() \
        .filter(lambda x: len(x[1]) > 1) \
        .map(lambda x: list(x[1])) \
        .flatMap(lambda x: generatePairs(x)) \
        .distinct() \
        .collect()
    print('length of candidates:', len(candidates)) # 19300205

    return candidates
    
    # verify candidates using Jaccard Similarity
    
    # output_file_path = './output/js_pairs.json'
    # with open(output_file_path, 'w', encoding='utf-8') as fp:
    #     for pair in res_valid_pairs:
    #         u1 = pair[0][0]
    #         u2 = pair[0][1]
    #         sim = pair[1]

    #         r_json = json.dumps({'u1': u1, 'u2': u2, 'sim': sim})
    #         fp.write(r_json)
    #         fp.write('\n')

def transformDataForUserBased(data_renamed):
    # data_renamed - [(uid, bid, star), ...]
    data_groupby_uid = data_renamed.map(lambda x: ((x[0], x[1]), [x[2]])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: averageRating(x)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: len(x[1]) > 2) \
        .map(lambda x: (x[0], convertToDict(x[1]))) \
        .collect()

    print('data_groupby_uid[:5]:', data_groupby_uid[:5])
    # print(len(data_groupby_uid)) # 26178

    return data_groupby_uid

def computePearsonCorrelationAndJaccardSimilarityUserBased(data, candidates):
    # data - [(uid, {bid: star, ...}), ...]
    print('data[:3]:', data[:3])
    data_dict = {} # {uid: (uid, {bid: star, ...}), ...}
    for uid, d in data:
        data_dict[uid] = (uid, d)
    # uid_list = list(data_dict.keys())
    uid_v = []
    for u1, u2 in candidates:
        pc1 = data_dict.get(u1)
        pc2 = data_dict.get(u2)
        if pc1 == None or pc2 == None:
            # if pc1 == None:
            #     print('None value:', u1)
            # else:
            #     print('None value:', u2)
            continue
        res = pc(data_dict[u1], data_dict[u2])
        if res != None:
            bids1 = list(data_dict[u1][1].keys())
            bids2 = list(data_dict[u2][1].keys())
            js_ = computeJaccardSimilarity(bids1, bids2)
            if js_ >= JACCARD_SIMILARITY_THRESHOLD:
                d_ = {'u1': res[0], 'u2': res[1], 'sim': res[2]}
                # print(d_)
                uid_v.append(d_)

    print('uid_v[:5] =', uid_v[:5])
    print('len(uid_v) =', len(uid_v)) # 542141
    return uid_v

def outputModelToFileUserBased(model, rename):
    # model - [{'u1': uid, 'u2': uid, 'sim': w}, ...]
    with open(model_file, 'w', encoding='utf-8') as fp:
        for item in model:
            item['u1'] = rename.getOriginalValue(item['u1'])
            item['u2'] = rename.getOriginalValue(item['u2'])
            fp.write(json.dumps(item))
            fp.write('\n')

if __name__ == "__main__":
    t_start = time.time()
    # define spark env
    conf = SparkConf() \
        .setAppName("task3train") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)

    # get data
    raw_data = getData(sc)

    if cf_type == 'item_based':
        # transform to generate a dataset for item-based model
        data_groupby_bid = transformDataForItemBased(raw_data)

        # compute Pearson Correlation w
        model = computePearsonCorrelationItemBased(data_groupby_bid)
        # in model, in (b1, b2) pairs, b1 < b2

        # output model to file
        outputModelToFile(model)
    elif cf_type == 'user_based':
        # get rename data
        data_renamed, bid_rename, uid_rename = getRenameData(raw_data)

        # find most possible user pairs using Min-Hash & LSH
        candidates = getCandidates(data_renamed, bid_rename, uid_rename)

        # transform to generate a dataset for user-based model
        data_groupby_uid = transformDataForUserBased(data_renamed)

        # compute Pearson Correlation w
        model = computePearsonCorrelationAndJaccardSimilarityUserBased(data_groupby_uid, candidates)

        # name back & output model to file
        outputModelToFileUserBased(model, uid_rename)
    else:
        print('wrong cf_type value.')
    
    t_end = time.time()
    print('time: %fs' % (t_end-t_start))