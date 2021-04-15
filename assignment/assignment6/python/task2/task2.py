"""
spark-submit task2.py <port #> <output_file_path>
"""

import sys
import time
import json
import random
import binascii
# import platform
from math import inf

from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.streaming import StreamingContext

import os
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

LARGE_NUMBER = inf
HOST_NAME = 'localhost'
WINDOW_LENGTH = 30
SLIDING_INTERVAL = 10

RANDOM_SEED = 100
EQUAL_THRESHOLD = 0.001
NUM_HASH = 30
BITS_LENGTH = 20
NOF_HASH_BUCKETS = 2 ** BITS_LENGTH - 1
NOF_SMALL_GROUPS = 10

port_num = int(sys.argv[1])
output_file_path = sys.argv[2]

# system_type = platform.system()
# if system_type == 'Linux':
#     print(system_type)
#     # for run on vocareum
#     import os
#     os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
#     os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

#     port_num = int(sys.argv[1])
#     output_file_path = sys.argv[2]
# elif system_type == 'Darwin':
#     print(system_type)
#     # run for local macos
#     port_num = 9999
#     output_file_path = "../output/output_task2.csv"
# else:
#     print('wrong system type.')
#     sys.exit()

class OneDimensionIntKMeans:
    def __init__(self, k_cluster, seed=RANDOM_SEED):
        self.k_cluster = k_cluster
        self.seed=seed
        
    def initialCentroids(self, d, seed=RANDOM_SEED):
        # d: dict - {idx: value, ...}
        random.seed(seed)
        centroids = {}
        distance = {}
        for i in range(self.k_cluster):
            if i == 0:
                first_centroid_tag = random.choice(list(d.keys()))
                centroids[first_centroid_tag] = d[first_centroid_tag]
                for idx in d:
                    distance[idx] = self.computeDistance(d[idx], d[first_centroid_tag])
            else:
                largest_one = [None, -1]
                for k in distance:
                    if distance[k] > largest_one[1]:
                        largest_one = [k, distance[k]]
                next_centroid_tag = largest_one[0]
                centroids[next_centroid_tag] = d[next_centroid_tag]
                for idx in d:
                    new_distance = self.computeDistance(d[idx], d[next_centroid_tag])
                    distance[idx] = min(distance[idx], new_distance)
        self.initial_centroids = centroids
        return centroids
                    
    def computeDistance(self, i, c):
        return abs(i - c)
            
    def clusterPoints(self, d, centroids):
        clusters = {}
        for idx in d:
            min_distance = [0, LARGE_NUMBER]
            for tag in centroids:
                dis = self.computeDistance(d[idx], centroids[tag])
                if dis < min_distance[1]:
                    min_distance = [tag, dis]
            new_tag = min_distance[0]
            if clusters.get(new_tag) == None:
                clusters[new_tag] = [idx]
            else:
                clusters[new_tag].append(idx)
        return clusters
    
    def computeCentroids(self, d, clusters):
        new_centroids = {}
        for tag in clusters:
            cluster = clusters[tag]
            new_centroid_value = sum([d[idx] for idx in cluster]) / len(cluster)
            new_centroids[tag] = new_centroid_value
        return new_centroids
    
    def checkCentroidsChanged(self, new_centroids, old_centroids, equal_threshold=EQUAL_THRESHOLD):
        if old_centroids == None:
            return True
        if len(old_centroids) != len(new_centroids):
            return True
        sum_all = 0
        for k in new_centroids:
            sum_all += abs(new_centroids[k] - old_centroids[k])
        dis_avg = sum_all / len(new_centroids)
        # print(dis_avg)
        if dis_avg > equal_threshold:
            return True
        else:
            return False

    def fit(self, d):
        old_centroids = None
        new_centroids = self.initialCentroids(d, self.seed)
        go_on = True
        while(go_on):
            clusters = self.clusterPoints(d, new_centroids)
            old_centroids = new_centroids
            new_centroids = self.computeCentroids(d, clusters)
            go_on = self.checkCentroidsChanged(new_centroids, old_centroids, EQUAL_THRESHOLD)
            # print('go_on:', go_on)
        return(clusters)
        
def convertStrToInt(s):
    return int(binascii.hexlify(s.encode('utf8')), 16)

def generateHashs(m, num_hash, seed=RANDOM_SEED):
    """
    m - the number of the hash buckets
    num_hash - the number of hash functions
    """
    def hashGenerator(i):
        a = a_s[i]
        b = b_s[i]
        def hashfunc(x):
            return (a * x + b) % m
        return hashfunc
    
    ab = set([])
    random.seed(seed)
    while(len(ab) < 2 * num_hash):
        ab.add(random.randint(1, 10 * num_hash))
        
    a_s = []
    for i in range(num_hash):
        a_s.append(ab.pop())
    b_s = list(ab)

    hash_functions = []
    for i in range(num_hash):
        hash_functions.append(hashGenerator(i))
    return hash_functions

def countTrailingZeros(n, bits_length):
    # n - int
    if n == 0:
        return 0 # bits_length
    count = 0
    while(n & 1 ^ 1):
        count += 1
        n = n >> 1
    return count

def groupsAvg(l, num_groups):
    # l: list - [2, 128, 64, ...]
    # num_groups: int - number of groups
    l_l = len(l)
    g_l = int(l_l / num_groups)
    i = 0
    groups_avg = []
    while(1):
        if g_l*(i+1) <= l_l: 
            group = l[g_l*i:g_l*(i+1)]
            group_avg = sum(group) / g_l
            groups_avg.append(group_avg)
            i += 1
        else:
            break
    return groups_avg

def getMedian(l):
    l = sorted(l)
    length_l = len(l)
    if length_l % 2 == 0:
        # even
        return (l[int(length_l/2)-1] + l[(int(length_l/2))]) / 2
    else:
        return (l[int((length_l-1)/2)])

def processRDD(rdd, hashs):
    t_cur = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    ground_truth = rdd.distinct().count()
    # print("ground_truth:", ground_truth)

    # # cluster powers
    # powers = rdd.map(lambda x: convertStrToInt(x)) \
    #     .map(lambda x: [h(x) for h in hashs]) \
    #     .map(lambda x: [countTrailingZeros(i, BITS_LENGTH) for i in x]) \
    #     .flatMap(lambda x: [(i, [x[i]]) for i in range(len(x))]) \
    #     .reduceByKey(lambda x, y: x + y) \
    #     .map(lambda x: max(x[1])) \
    #     .collect()

    # powers_dict = dict([(i, powers[i]) for i in range(len(powers))])

    # km = OneDimensionIntKMeans(NOF_SMALL_GROUPS)
    # clusters = km.fit(powers_dict)

    # avg_groups = []
    # l_for_print = []
    # for k in clusters:
    #     group = clusters[k]
    #     group_avg = sum([2**powers_dict[x] for x in group]) / len(group)
    #     avg_groups.append(group_avg)
    #     l_for_print.append([2**powers_dict[x] for x in group])

    # # cluster 2 to the powers
    # powers = rdd.map(lambda x: convertStrToInt(x)) \
    #     .map(lambda x: [h(x) for h in hashs]) \
    #     .map(lambda x: [countTrailingZeros(i, BITS_LENGTH) for i in x]) \
    #     .flatMap(lambda x: [(i, [x[i]]) for i in range(len(x))]) \
    #     .reduceByKey(lambda x, y: x + y) \
    #     .map(lambda x: max(x[1])) \
    #     .collect()

    # es_dict = dict([(i, 2**powers[i]) for i in range(len(powers))])

    # km = OneDimensionIntKMeans(NOF_SMALL_GROUPS)
    # clusters = km.fit(es_dict)

    # avg_groups = []
    # l_for_print = []
    # for k in clusters:
    #     group = clusters[k]
    #     group_avg = sum([es_dict[x] for x in group]) / len(group)
    #     avg_groups.append(group_avg)
    #     l_for_print.append([es_dict[x] for x in group])

    # estimation = getMedian(avg_groups)
    # estimation = round(estimation) # python round bug



    es = rdd.map(lambda x: convertStrToInt(x)) \
        .map(lambda x: [h(x) for h in hashs]) \
        .map(lambda x: [countTrailingZeros(i, BITS_LENGTH) for i in x]) \
        .flatMap(lambda x: [(i, [x[i]]) for i in range(len(x))]) \
        .reduceByKey(lambda x, y: x + y) \
        .map(lambda x: 2**max(x[1])) \
        .collect()
    # es: list - [2, 128, 64, ...]
    avg_groups = groupsAvg(es, NOF_SMALL_GROUPS)
    estimation = getMedian(avg_groups)
    estimation = round(estimation) # python round bug

    # print(t_cur, ground_truth, estimation)
    # print(es, '\n')

    
    # print(l_for_print, '\n')
    with open(output_file_path, 'a', encoding='utf-8') as fp:
        fp.write("%s,%d,%d\n" % (t_cur, ground_truth, estimation))




def main():
    hashs = generateHashs(NOF_HASH_BUCKETS, NUM_HASH)
    with open(output_file_path, 'w', encoding='utf-8') as fp:
        fp.write("Time,Ground Truth,Estimation\n")

    conf = SparkConf() \
        .setAppName("task") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    scc = StreamingContext(sc, 5)

    window_stream_rdd = scc.socketTextStream(HOST_NAME, port_num) \
        .window(WINDOW_LENGTH, SLIDING_INTERVAL) \
        .map(json.loads) \
        .map(lambda x: x['city']) \
        .filter(lambda x: x != "") \
        .cache()
    
    window_stream_rdd.foreachRDD(lambda rdd: processRDD(rdd, hashs))



    scc.start()
    scc.awaitTermination()

if __name__ == "__main__":
    t_s = time.time()
    try:
        main()
    finally:
        print('Duration: %fs.' % (time.time()-t_s))

# if __name__ == "__main__":
#     main()
