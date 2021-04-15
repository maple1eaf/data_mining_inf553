"""
spark-submit task1.py <input_file> <output_file>

spark-submit task1_dev.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw3/data/train_review.json" "./output/task1_dev_output.json"
"""

import sys
import time
import json
import random

from pyspark import SparkConf, SparkContext


# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

NUM_HASHS = 2
LARGE_NUMBER = 999999
NUM_BANDS = 2
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

# def generateHashs(n, m):
#     """
#     generate a list of hash functions
#     :n - number of hash functions we want to generate
#     :m - number of attributes
#     """
#     random.seed(1208)
#     ab = []
#     for i in range(2*n):
#         n1 = random.randint(1,m)
#         if n1 not in ab:
#             ab.append(n1)
#     a_list = ab[:n]
#     b_list = ab[n:]
#     m_new = smallestPrime(m)
    
#     print('ab list:', ab)
#     print('the prime number m_new:', m_new)

#     def generateHash(i):
#         a = a_list[i]
#         b = b_list[i]
#         def hashfunc(x):
#             return (a * x + b) % m_new
#         return hashfunc
#     return [generateHash(i) for i in range(n)]

def generateHashs(n, m):
    def hash1(x):
        return (x+1)%5
    def hash2(x):
        return (3*x+1)%5
    return [hash1, hash2]

# class Signature():
#     def __init__(self, l):
#         """
#         l: list
#         """
#         self.length = len(l)
#         self.value = l

#     def get(self):
#         return self.value

#     def set(self, new_list):
#         self.value = new_list

#     def update(self, other):
#         current_ = self.value
#         new_ = other.value
#         for i in range(self.length):
#             if new_[i] < current_[i]:
#                 current_[i] = new_[i]

def updateSignature_reduce(s_cur, s_new, length):
    l_ = [0] * length
    for i in range(length):
        if s_new[i] < s_cur[i]:
            l_[i] = s_new[i]
        else:
            l_[i] = s_cur[i]
    return l_

def updateSignature(s_cur, s_new, length):
    for i in range(length):
        if s_new[i] < s_cur[i]:
            s_cur[i] = s_new[i]

def minHash(partition, b_pt, u_pt, n_hashs):
    n_bins = u_pt.length
    hashs = generateHashs(n_hashs, n_bins)
    sm = {}
    for b in b_pt.indice:
        sm[b] = [LARGE_NUMBER] * n_hashs

    for (b, u) in partition:
        new_signature = [h_func(u) for h_func in hashs]
        updateSignature(sm[b], new_signature, n_hashs)
    print('were done here.')
    sm_partition = [(k, sm[k]) for k in b_pt.indice]
    # print('---sm_partition---:', sm_partition[:20])
    return sm_partition

def minHashPar(partition, hashs, b_pt):
    # build a signature matrix for a partition
    sm = {}
    for b in b_pt.indice:
        sm[b] = [LARGE_NUMBER] * NUM_HASHS
    # to get new hash values for each row
    hash_values = [0] * NUM_HASHS

    for r in partition:
        print('r---', r)
        u_id = r[0] # u_id also represents to #row
        b_ids = r[1] # a list of business id
        for i in range(NUM_HASHS):
            hash_values[i] = hashs[i](u_id)
        print('hash_values---', hash_values)
        for b in b_ids:
            s_cur = sm[b]
            s_new = hash_values
            s_length = NUM_HASHS
            updateSignature(s_cur, s_new, s_length)
    sm_list = [(k, sm[k]) for k in b_pt.indice]
    print('done a signature matrix / partition.')
    return sm_list




def hashForLSH(l):
    l_len = len(l)
    random.seed(328)
    l_ = []
    for i in range(l_len):
        a = random.randint(1, 10000)
        if a not in l_:
            l_.append(a)
    sum_ = 0
    for i in range(l_len):
        sum_ = sum_ + l_[i] * l[i]
    return sum_ % 500000

def splitIntoBandsAndHash(x, b):
    """
    x - (b_id, [100 of ints])
    """
    b_id = x[0]
    vs = x[1]
    r = int(NUM_HASHS / b)
    res = []
    for i in range(b):
        v_hash = hashForLSH(vs[i*r:(i+1)*r])
        res.append(((v_hash, i), b_id))
    return res

def generatePairs(l):
    l.sort()
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

def countIntersect(ite, candidates):
    l = list(ite)
    l_set = set(l)
    return [(candidate, 1) for candidate in candidates if set(candidate).issubset(l_set)]

def countJaccard(x, lu_dict):
    candi = x[0]
    inter_count = x[1]
    return inter_count / (lu_dict[candi[0]] + lu_dict[candi[1]] - inter_count)


def getData(sc):
    """
    :return data - [(b_index, u_index), ...]
    """
    raw_data = sc.textFile(input_file_path) \
        .map(json.loads) \
        .map(lambda r: (r['business_id'], r['user_id'])) \
        .distinct() \
        .cache()

    # generate a rename table for business_id
    bussiness_ids_distinct = raw_data.map(lambda x: x[0]).distinct().collect()
    # bussiness_ids_distinct.sort()
    bussiness_ids_pt = ProjectionTable(bussiness_ids_distinct)
    # print(bussiness_ids_pt.iv[-20:])

    # generate a rename table for business_id
    user_ids_distinct = raw_data.map(lambda x: x[1]).distinct().collect()
    user_ids_distinct.sort()
    user_ids_pt = ProjectionTable(user_ids_distinct)
    # print(user_ids_pt.iv[-20:])

    # data - [(b_id, u_id), ...]
    data = raw_data.map(lambda x: (bussiness_ids_pt.getIndex(x[0]), user_ids_pt.getIndex(x[1]))) \
        .cache()

    # data_groupby_uid - [(u_id, [b_id1, b_id2, ...]), ...]
    data_groupby_uid = data.map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .cache()

    return data, bussiness_ids_pt, user_ids_pt, data_groupby_uid

def process1(data, b_pt, u_pt, NUM_HASHS):
    sm = data.mapPartitions(lambda x: minHash(x, b_pt, u_pt, NUM_HASHS)) \
        .reduceByKey(lambda x, y: updateSignature_reduce(x, y, NUM_HASHS)) \
        .sortByKey() \
        .cache()

    result_sm = sm.collect()
    t_sm = time.time()
    print('done signature matrix. time:%fs.' % (t_sm - t_a))
    print('signature matrix whole:', result_sm[:2])
    print('length of signature matrix:', len(result_sm))

    return sm

def generateSignatureMatrix(data_groupby_uid, hashs, b_pt):
    sm_partition = data_groupby_uid.mapPartitions(lambda p: minHashPar(p, hashs, b_pt))
    sm = sm_partition.reduceByKey(lambda x, y: updateSignature_reduce(x, y, NUM_HASHS)) \
        .sortByKey() \
        .cache()

    result_sm = sm.collect()
    t_sm = time.time()
    print('done signature matrix. time:%fs.' % (t_sm - t_a))
    print('signature matrix whole:', result_sm)
    print('length of signature matrix:', len(result_sm))
    with open('./output/signature_matrix.json', 'w', encoding='utf-8') as fp:
        for row in result_sm:
            len_ = len(row[1])
            fp.write(str(len_))
            fp.write('  ')

            row_0 = b_pt.getValue(row[0])
            fp.write(str(row_0))
            fp.write('  ')

            row_json = json.dumps(row)
            fp.write(row_json)

            fp.write('\n')

    return sm

def LSH(sm, b):
    candidates = sm.flatMap(lambda x: splitIntoBandsAndHash(x, b)) \
        .groupByKey() \
        .filter(lambda x: len(x[1]) > 1) \
        .map(lambda x: list(x[1])) \
        .flatMap(lambda x: generatePairs(x)) \
        .distinct() \
        .collect()
        # .filter(lambda x: len(x[1])>1)
    print(candidates[:20])
    print('length of candidates:',len(candidates))
    return candidates

def process3(data, candidates):
    # # 1
    # length_of_users = data.map(lambda x: (x[0], 1)) \
    #     .reduceByKey(lambda x, y: x + y) \
    #     .collect()
    # print('---length_of_users:---', length_of_users[:20])

    # lu_dict = {}
    # for (k,v) in length_of_users:
    #     lu_dict[k] = v
    
    # # 2
    # count_of_candidate_intersect = data.map(lambda x: (x[1], x[0])) \
    #     .groupByKey() \
    #     .flatMap(lambda x: countIntersect(x[1], candidates)) \
    #     .reduceByKey(lambda x, y: x + y) \
    #     .map(lambda x: (x[0],countJaccard(x, lu_dict))) \
    #     .filter(lambda x: x[1]>=0.05) \
    #     .collect()
    # # print('---length_of_intersect---', count_of_candidate_intersect[:20])

    # no spark
    t_p4 = time.time()
    data_groupby_uid = data.groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .collect()
    u_dict = {}
    for (k,v) in data_groupby_uid:
        u_dict[k] = v

    res_valid_pairs = []
    for (b1, b2) in candidates:
        js_ = jaccardSimilarity(u_dict[b1], u_dict[b2])
        if js_ >= JACCARD_SIMILARITY_THRESHOLD:
            res_valid_pairs.append(((b1, b2), js_))
    
    print('---number of valid pairs:---', len(res_valid_pairs))
    t_p5 = time.time()
    print('---done count jaccard similarity. time:%fs.---' % (t_p5 - t_p4))

    return res_valid_pairs


def process4(js_pairs, b_pt):
    res = []
    for pair in js_pairs:
        # pair - [[b1, b2], sim]
        b1 = b_pt.getValue(pair[0][0])
        b2 = b_pt.getValue(pair[0][1])
        sim = pair[1]
        res.append({'b1': b1, 'b2': b2, 'sim': sim})
    
    with open(output_file_path, 'w', encoding='utf-8') as fp:
        for r in res:
            r_json = json.dumps(r)
            fp.write(r_json)
            fp.write('\n')


if __name__ == "__main__":
    t_a = time.time()
    # define spark env
    conf = SparkConf() \
        .setAppName("task1") \
        .setMaster("local[*]")
    sc = SparkContext(conf=conf)

    data = sc.textFile(input_file_path) \
        .map(json.loads) \
        .map(lambda r: (r['business_id'], r['user_id'])) \
        .filter(lambda x: x[0]!=None and x[1]!=None)\
        .distinct() \
        .count()

    # data = sc.textFile(input_file_path) \
    #     .map(json.loads) \
    #     .map(lambda r: (r['business_id'], [r['user_id']])) \
    #     .reduceByKey(lambda a, b: a+b if b[0] not in a else a) \
    #     .flatMap(lambda x: x[1]) \
    #     .count()
    print(data)

    # # step 1: get data and the rename tables
    # data, b_pt, u_pt, data_groupby_uid = getData(sc)
    # print('b_pt:', b_pt.iv)
    # print('u_pt:', u_pt.iv)

    # # generate a list of hash functions for Min-Hash
    # hashs = generateHashs(NUM_HASHS, u_pt.length)

    # # step 2: implement Min-Hash to transfer user_id to signature
    # # sm = process1(data, b_pt, u_pt, NUM_HASHS)
    # sm = generateSignatureMatrix(data_groupby_uid, hashs, b_pt)

    # # step 3: implement LSH to generate candidates
    # candidates = LSH(sm, NUM_BANDS)

    # # step 4: verify candidates using Jaccard similarity
    # js_pairs = process3(data, candidates)

    # # step 5: output
    # process4(js_pairs, b_pt)
    # # with open(output_file_path, 'w', encoding='utf-8') as fp:
    # #     json.dump(js_pairs, fp)

    t_b = time.time()
    print('time consume: %fs' % (t_b-t_a))