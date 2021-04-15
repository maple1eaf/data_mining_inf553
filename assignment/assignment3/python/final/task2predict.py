"""
spark-submit task2predict.py <test_file> <model_file> <output_file>

spark-submit task2predict.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw3/data/test_review.json" "./output/task2.model" "./output/task2.predict"

vocareum:
python3 task2predict.py $ASNLIB/publicdata//test_review.json task2.model task2.predict
"""

import sys
import time
import json
import re
from math import sqrt

from pyspark import SparkConf, SparkContext, StorageLevel

# for run on vocareum
# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

test_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]

def toUserProfileBitmap(l, bp):
    """
    l - [bid1, bid2, ...]
    bp - bp - {bid: bitmap, ...}
    """
    bitmap_res = bp[l[0]]
    for i in range(1, len(l)):
        bitmap_i = bp[l[i]]
        bitmap_res = bitmap_res.unionTwo(bitmap_i)
    
    bitmap_res.countAndSetCountOne()
    return bitmap_res

def toBitmap(bid, bp):
    sca = bp.get(bid)
    if sca == None:
        return None
    size = sca[0]
    count_one = sca[1]
    array = sca[2]
    bit = Bitmap(1) # any number
    bit.setBitmap(size, count_one, array)
    return bit

def countOneInt(integer):
    count = 0
    while(integer != 0):
        if integer & 1 == 1:
            count += 1
        integer = integer >> 1
    return count

def countOne(l):
    sum = 0
    for i in range(len(l)):
        sum += countOneInt(l[i])
    return sum

def computeCosineSimilarity(up, bp):
    d1 = up[0]
    d2 = bp[0]
    intersect_list = [up[1][i] & bp[1][i] for i in range(len(up[1]))]
    n1 = countOne(intersect_list)
    cs = n1 / sqrt(d1 * d2)
    return cs

def compareSC(x, bps, ups):
    up = ups.get(x[0])
    bp = bps.get(x[1])
    # up, bp - [count_one, bit_array]
    if up == None or bp == None:
        return None
    cosine_similarity = computeCosineSimilarity(up, bp)
    if cosine_similarity >= 0.01:
        # print('get one!')
        return (x[0], x[1], cosine_similarity)
    else:
        return None

def getData(sc):
    # get model data
    with open(model_file, 'r', encoding='utf-8') as fp:
        model = json.load(fp)

    # get test data
    raw_data = sc.textFile(test_file).map(json.loads)
    # print('num of partitions:', raw_data.getNumPartitions())
    return model, raw_data

def modelParser(model):
    bitmap_size = model['bitmap_size']
    bps = model['bps']
    ups = model['ups']
    # bitmap_size - how many bits of bit_array
    # bps - {bid: [count_one, bit_array], ...}
    # ups - {uid: [count_one, bit_array], ...}

    return bps, ups, bitmap_size

def process(raw_data, bps, ups):
    pairs = raw_data.map(lambda r: (r['user_id'], r['business_id'])) \
        .map(lambda x: compareSC(x, bps, ups)) \
        .filter(lambda x: x != None) \
        .collect()
    # pairs - (uid, bid, cosine_similarity)
    return pairs

def writeOutputFile(uid_bid_cs):
    with open(output_file, 'w', encoding='utf-8') as fp:
        for item in uid_bid_cs:
            fp.write('{"user_id": "%s", "business_id": "%s", "sim": %s}' % (item[0], item[1], item[2]))
            fp.write('\n')

if __name__ == "__main__":
    t_start = time.time()
    # define spark env
    conf = SparkConf() \
        .setAppName("task2predict") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)

    # get data and model
    model, raw_data = getData(sc)

    # parse model
    bps, ups, bitmap_size = modelParser(model)

    # get users like list
    pairs = process(raw_data, bps, ups)
    # print('number of pairs:', len(pairs))
    
    # output result
    writeOutputFile(pairs)


    t_end = time.time()
    print('time: %fs' % (t_end-t_start))
