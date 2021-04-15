"""
spark-submit task1.py <input_file> <output_file>

spark-submit task1_dev.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw3/data/train_review.json" "./output/task1_dev_output.json"
"""

import sys
import time
import json
import random

from pyspark import SparkConf, SparkContext

# for run on vocareum
# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

NUM_HASHS = 30
LARGE_NUMBER = 999999
NUM_BANDS = NUM_HASHS
JACCARD_SIMILARITY_THRESHOLD = 0.05 # >=

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

class ProjectionTable():
    """
    key-value domains are one to one projection.
    """
    def __init__(self, values):
        """
        values - iterable
        iv - index-value pair
        """
        self.values = values
        self.length = len(values)
        self.indice = list(range(self.length))
        self.iv = list(enumerate(values))
    
    def getIndex(self, value):
        for i, v in self.iv:
            if v == value:
                return i
        return None
    
    def getValue(self, index):
        for i, v in self.iv:
            if i == index:
                return v
        return None

def isPrime(num):
    if num == 1:
        return False
    for divider in range(2, num):
        if num % divider == 0:
            return False
    return True

def smallestPrime(num):
    while(1):
        num = num + 1
        if isPrime(num):
            return num

def generateHashs(n, m):
    """
    generate a list of hash functions
    :n - number of hash functions we want to generate
    :m - number of attributes
    """
    random.seed(12)
    ab = []
    for i in range(2*n):
        n1 = random.randint(1,m)
        if n1 not in ab:
            ab.append(n1)

    a_list = ab[:n]
    b_list = ab[n:]
    m_new = smallestPrime(m)
    
    print('ab list length:', len(ab))
    print('the prime number m_new:', m_new)

    def generateHash(i):
        a = a_list[i]
        b = b_list[i]
        def hashfunc(x):
            return (a * hash(x) + b) % m_new
        return hashfunc
    return [generateHash(i) for i in range(n)]

def generateHashForLSH(r):
    random.seed(8)
    l_ = []
    for i in range(r):
        a = random.randint(1, 10000)
        if a not in l_:
            l_.append(a)

    def h(l):
        # l - list of integers
        l_len = len(l)
        sum_ = 0
        for i in range(l_len):
            sum_ = sum_ + l_[i] * l[i]
        return sum_ % 500000
    return h


def updateSignature(s_cur, s_new, length):
    for i in range(length):
        if s_new[i] < s_cur[i]:
            s_cur[i] = s_new[i]

def minHash(x, hashs):
    b_id = x[0]
    u_ids = x[1]
    signature = [LARGE_NUMBER] * NUM_HASHS
    for u in u_ids:
        s_ = [hashs[i](u) for i in range(NUM_HASHS)]
        updateSignature(signature, s_, NUM_HASHS)
    res = (b_id, signature)
    return res

# def minHash2(x, hashs):
#     b_id = x[0]
#     u_ids = x[1]
#     signature = []
#     for hafunc in hashs:
#         cur_ = LARGE_NUMBER
#         for u in u_ids:
#             hash_value = hafunc(u)
#             if hash_value < cur_:
#                 cur_ = hash_value
#         signature.append(cur_)
#     return(b_id, signature)

def LSH(x, b, r, hash_lsh):
    b_id = x[0]
    signature = x[1]
    ress = []
    for i in range(b):
        v_hash = hash_lsh(signature[i*r:(i+1)*r])
        res = ((v_hash, i), b_id)
        ress.append(res)
    return ress

def generatePairs(l):
    l_len = len(l)
    res_l = [(l[i],l[j]) for i in range(l_len) for j in range(i+1, l_len)]
    return res_l

def jaccardSimilarity(l1, l2):
    # items in l1 are unique to each other, so does l2
    l1_len = len(l1)
    l2_len = len(l2)
    intersect_set = set(l1).intersection(set(l2))
    inter_len = len(intersect_set)
    union_len = l1_len + l2_len - inter_len
    js = inter_len / union_len
    return js

def getData(sc):
    """
    :return data - [(b_index, u_index), ...]
    """
    raw_data = sc.textFile(input_file_path) \
        .map(json.loads) \
        .map(lambda r: (r['business_id'], r['user_id'])) \
        .distinct() \
        .cache()

    # count how many distinct business_id
    b_id_len = raw_data.map(lambda x: x[0]).distinct().count()

    # ount how many distinct user_id
    u_id_len = raw_data.map(lambda x: x[1]).distinct().count()

    # data - [(b_id, u_id), ...]
    data = raw_data.groupByKey().cache()

    return data, b_id_len, u_id_len

def process(data, hashs, b, r, hash_lsh):
    candidates = data.map(lambda x: minHash(x, hashs)) \
        .flatMap(lambda x: LSH(x, b, r, hash_lsh)) \
        .groupByKey() \
        .filter(lambda x: len(x[1]) > 1) \
        .map(lambda x: list(x[1])) \
        .flatMap(lambda x: generatePairs(x)) \
        .distinct() \
        .collect()

    t_candi = time.time()
    print('done candidates. time:%fs.' % (t_candi - t_a))
    # print('candidates:', candidates)
    print('length of candidates:', len(candidates))

    return candidates



def process3(data, candidates):
    # no spark
    t_p4 = time.time()
    data_groupby_bid = data.map(lambda x: (x[0], list(x[1]))).collect()
    b_dict = {}
    for (k,v) in data_groupby_bid:
        b_dict[k] = v

    res_valid_pairs = []
    for (b1, b2) in candidates:
        js_ = jaccardSimilarity(b_dict[b1], b_dict[b2])
        if js_ >= JACCARD_SIMILARITY_THRESHOLD:
            res_valid_pairs.append(((b1, b2), js_))
    
    print('---number of valid pairs:---', len(res_valid_pairs))
    t_p5 = time.time()
    print('---done count jaccard similarity. time:%fs.---' % (t_p5 - t_p4))

    return res_valid_pairs


def process4(true_positives):
    with open(output_file_path, 'w', encoding='utf-8') as fp:
        for pair in true_positives:
            b1 = pair[0][0]
            b2 = pair[0][1]
            sim = pair[1]

            r_json = json.dumps({'b1': b1, 'b2': b2, 'sim': sim})
            fp.write(r_json)
            fp.write('\n')


if __name__ == "__main__":
    t_a = time.time()
    # define spark env
    conf = SparkConf() \
        .setAppName("task1") \
        .setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # step 1: get data and the rename tables
    data, b_id_len, u_id_len = getData(sc)

    # generate a list of hash functions for Min-Hash
    hashs = generateHashs(NUM_HASHS, u_id_len)
    b = NUM_BANDS
    r = int(NUM_HASHS / NUM_BANDS)
    hash_for_lsh = generateHashForLSH(r)

    # step 2: implement Min-Hash to transfer user_id to signature
    candidates = process(data, hashs, b, r, hash_for_lsh)

    # step 4: verify candidates using Jaccard similarity
    true_positives = process3(data, candidates)

    # step 5: output
    process4(true_positives)
    # with open(output_file_path, 'w', encoding='utf-8') as fp:
    #     json.dump(js_pairs, fp)

    t_b = time.time()
    print('time consume: %fs' % (t_b-t_a))