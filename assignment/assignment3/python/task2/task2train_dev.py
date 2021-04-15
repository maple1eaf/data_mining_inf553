"""
spark-submit task2train.py <train_file> <model_file> <stopwords>

spark-submit task2train_dev.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw3/data/train_review.json" "./output/task2model_dev.json" "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw3/data/stopwords"
"""

import sys
import time
import json
import re
from math import log

from pyspark import SparkConf, SparkContext, StorageLevel
from task2support import *

# for run on vocareum
# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

train_file = sys.argv[1]
model_file = sys.argv[2]
stopwords_file = sys.argv[3]

def removePucWords(t, words):
    """
    remove all the punctures and numbers and stopwords
    :t - the input text
    :words - the words which we wants to remove, stopwords, most rare words, most frequent words...
    """
    t = t.lower()
    pattern = r"[a-z]+" # only words
    words_without_punc_num = re.findall(pattern, t)
    words_clean = tuple([word for word in words_without_punc_num if word not in words])
    # print('length of words_clean1:', len(words_clean))
    return words_clean

def countTFIDF(x, max_count):
    """
    x - (word, ([(bid, cib), ...], idf))
    max_count - {bid: max_count, ...}
    """
    word = x[0]
    counts = x[1][0]
    idf = x[1][1]

    l_ = []
    for item in counts:
        bid = item[0]
        cib = item[1]
        tf = cib / max_count[bid]
        tf_idf = tf * idf
        l_.append((bid, [(word, tf_idf)]))

    return l_

def top200(l):
    l.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in l[:200]]

def transferToBitmap(l, length_list, all_words_dict):
    bitmap = Bitmap(length_list)
    bitmap.initialize(l, all_words_dict)
    return bitmap

def createUserProfileBitmap(l, bps):
    """
    l - [bid1, bid2, ...]
    bp - bp - {bid: bitmap, ...}
    """
    bitmap_res = bps[l[0]]
    for i in range(1, len(l)):
        bitmap_i = bps[l[i]]
        bitmap_res = bitmap_res.unionTwo(bitmap_i)
    bitmap_res.countAndSetCountOne()
    return bitmap_res


def getData(sc):
    raw_data = sc.textFile(train_file) \
        .map(json.loads)

    data_concatenated = raw_data.map(lambda x:(x['business_id'], x['text'])) \
        .reduceByKey(lambda x, y: x + ' ' + y)

    # get stopwords
    stopwords = sc.textFile(stopwords_file).collect()

    # get total number of docs. 10253
    num_docs = data_concatenated.count()
    print('number of docs:', num_docs)
    
    return raw_data, data_concatenated, stopwords, num_docs

def generateBusinessProfile(data_concatenated, stopwords, num_docs):
    words_remove_punc_num_sw = data_concatenated.map(lambda x: (x[1], x[0])) \
        .map(lambda x: (removePucWords(x[0], stopwords), x[1]))
        # .persist(StorageLevel.MEMORY_AND_DISK)

    # iterator-> ((word, b_id), (1, 1))
    # first 1 for count, second 1 for tag occurence
    word_bid_count_occurence = words_remove_punc_num_sw.flatMap(lambda x: [((w, x[1]), (1, 1)) for w in x[0]]) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    # 34561473
    count_words_total = word_bid_count_occurence.count()
    print('number of words total:', count_words_total)

    # 34
    rare_word_threshold = int(count_words_total * 0.000001)

    # distinct words: 159230 -> 21259 after filtering
    candidate_words = word_bid_count_occurence.map(lambda x: (x[0][0], x[1][0])) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] > rare_word_threshold) \
        .map(lambda x: x[0]) \
        .collect()
    print(len(candidate_words))

    t_1 = time.time()
    # plan 1: after reduce -> 9943698 then after filter -> 9500760, filter need 340s
    # after map -> (word, ([(bid, count_in_bid)], occur))
    # plan 2: after double reduce -> 159230 (54s) then after filter 21259 (65s), filter need 11s
    # data_after_filter -> (word, ([(bid, cib), ...], occurs))
    data_after_filter = word_bid_count_occurence.reduceByKey(lambda x, y: (x[0]+y[0], 1)) \
        .map(lambda x: (x[0][0], ([(x[0][1], x[1][0])], x[1][1]))) \
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
        .filter(lambda x: x[0] in candidate_words)
    t_2 = time.time()
    print('data_after_filter time: %fs' % (t_2-t_1))

    # data_with_idf -> (word, ([(bid, cib), ...], idf))
    data_with_idf = data_after_filter.map(lambda x: (x[0], (x[1][0], log(num_docs / x[1][1], 2)))) \
        .persist(StorageLevel.MEMORY_AND_DISK)
    # word_bid_count_occurence.unpersist()
    

    max_count_each_bid = data_with_idf.flatMap(lambda x: x[1][0]) \
        .reduceByKey(lambda x, y: max(x, y)) \
        .collect()
    max_count = {}
    for (bid, count) in max_count_each_bid:
        max_count[bid] = count
    
    print(max_count_each_bid[:5])
    print('length max count for each bid:', len(max_count_each_bid))
    t_3 = time.time()
    print('max_count_each_bid time: %fs' % (t_3-t_2))

    bps = data_with_idf.flatMap(lambda x: countTFIDF(x, max_count)) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], top200(x[1]))) \
        .collect()
    print('business_profiles:', bps[:3])
    t_4 = time.time()
    print('business_profile_s takes time:%fs.' % (t_4-t_3))

    return bps

def convertToBitmap(bps_list, sc):
    all_words = sc.parallelize(bps_list) \
        .flatMap(lambda x: x[1]) \
        .distinct() \
        .collect()
    length_all_words = len(all_words)
    print('all_words:', all_words[:100])
    print('length of all_words:', length_all_words)
    all_words_dict = {}
    for i in range(len(all_words)):
        all_words_dict[all_words[i]] = i
    
    bps_bitmap = sc.parallelize(bps_list) \
        .map(lambda x: (x[0], transferToBitmap(x[1], length_all_words, all_words_dict))) \
        .collect()
    
    print(bps_bitmap[:2])

    bps = {}
    for (bid, bitmap) in bps_bitmap:
        bps[bid] = bitmap

    # print('test:', business_profile_s_bitmap_dict['bZMcorDrciRbjdjRyANcjA'])
    # model = {}
    # model['words_series'] = all_words
    # model['words_dict'] = all_words_dict
    # model['business_profiles_bitmap_dict'] = business_profile_s_bitmap_dict

    return bps, length_all_words

def generateUserProfilesBitmap(bps, raw_data):
    ups_list = raw_data.map(lambda r: (r['user_id'], r['business_id'])) \
        .distinct() \
        .map(lambda x: (x[0], [x[1]])) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: (x[0], createUserProfileBitmap(x[1], bps))) \
        .collect()
    ups = {}
    for uid, bitmap in ups_list:
        ups[uid] = bitmap
    return ups

def createModel(bps, ups, bitmap_size):
    model = {}
    model['bitmap_size'] = bitmap_size

    bps_model = {}
    for bid in bps:
        bitmap = bps[bid]
        bps_model[bid] = (bitmap.count_one, bitmap.array)
    
    ups_model = {}
    for uid in ups:
        bitmap = ups[uid]
        ups_model[uid] = (bitmap.count_one, bitmap.array)

    model['bps'] = bps_model
    model['ups'] = ups_model

    return model

def outputModelToFile(model):
    with open(model_file, 'w', encoding='utf-8') as fp:
        json.dump(model, fp)

if __name__ == "__main__":
    t_start = time.time()
    # define spark env
    conf = SparkConf() \
        .setAppName("task2train") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)

    # get data
    raw_data, data_concatenated, stopwords, num_docs = getData(sc)

    bps_list = generateBusinessProfile(data_concatenated, stopwords, num_docs)

    # convert business profiles to bitmap
    bps, bitmap_size = convertToBitmap(bps_list, sc)

    # create user profiles
    ups = generateUserProfilesBitmap(bps, raw_data)

    # build and output the model
    model = createModel(bps, ups, bitmap_size)

    # output model to file
    outputModelToFile(model)
    
    t_end = time.time()
    print('time: %fs' % (t_end-t_start))