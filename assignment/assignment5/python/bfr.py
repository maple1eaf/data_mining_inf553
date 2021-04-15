"""
python3 bfr.py <input_path> <n_cluster> <out_file1> <out_file2>

spark-submit bfr.py "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw5/data/test2" 10 "./out_file1.json" "./out_file2.csv"

on vocareum:
python3 bfr.py $ASNLIB/publicdata/test5 10 out_file1.json out_file2.csv
"""

import json
import os
import random
import sys
import time
from math import sqrt

from pyspark import SparkConf, SparkContext, StorageLevel

import platform
system_type = platform.system()
if system_type == 'Linux':
    print(system_type)
    # for run on vocareum
    os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
elif system_type == 'Darwin':
    print(system_type)
    # run for local macos
    os.environ['PYSPARK_PYTHON'] = '/Users/markduan/opt/anaconda3/envs/inf553/bin/python'
    os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/markduan/opt/anaconda3/envs/inf553/bin/python'

    # input_path = "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw5/data/test3"
    # n_cluster = 5
    # out_file1 = "./out_file1.json"
    # out_file2 = "./out_file2.csv"
else:
    print('wrong system type.')
    sys.exit()

SMALL_SAMPLE_RATIO = 0.1
RANDOM_SEED = 1208
LARGE_NUMBER = sys.maxsize * 1.0
EQUAL_THRESHOLD = 0.01
ALPHA_FOR_MD = 4
MULTI_FOR_FIRST_ROUND = 3
MULTI_FOR_OTHER_ROUND = 3
KMEANS_MAX_ITER = 15

input_path = sys.argv[1]
n_cluster = int(sys.argv[2])
out_file1 = sys.argv[3]
out_file2 = sys.argv[4]

def getSparkEnv():
    conf = SparkConf() \
        .setAppName("task") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    return sc

def getFilesPath(input_path):
    files = os.listdir(input_path)
    files_start_with_slash = [os.path.join(input_path, f) for f in files]
    files_start_with_file = ['file://' + f for f in files_start_with_slash]
    files_start_with_file.sort()
    print('%d files.' % (len(files_start_with_file)))
    return files_start_with_file

def loadFileRDD(sc, file_path):
    """
    :return rdd - [(idx, (attrs...)), ...] idx: int attr: float
    """
    data_points_rdd = sc.textFile(file_path) \
        .map(lambda x: x.split(',')) \
        .map(lambda x: (int(x[0]), tuple(x[1:]))) \
        .mapValues(lambda v: tuple([float(x) for x in v]))
    return data_points_rdd

def splitData(data_points, ratio, random_seed=1208):
    # randomly select samples and the rest data
    random.seed(random_seed)
    random.shuffle(data_points)
    num_data = len(data_points)
    num_small_sample = int(num_data * ratio)
    small_sample = data_points[:num_small_sample]
    rest_data = data_points[num_small_sample:]
    return small_sample, rest_data

def computeDistanceForEachNode(x, new_centroid):
    # x - (idx, (attrs...))
    # new_centroid - (cluster_tag, (attrs...))
    dis = computeDistance(x[1], new_centroid[1])
    return (x[0], dis) # (idx, dis)

def getInitialCentroids(data_rdd, size, method="random"):
    if method == "random":
        # random
        centroids_list = data_rdd.map(lambda x: x[1]) \
            .takeSample(False, size, RANDOM_SEED)
        centroids = dict(centroids_list)
        # the key of each centroid is the tag of the cluster
        # {cluster_tag: (attrs...), ...}
        return centroids
    elif method == "kmeans++":
        t_1 = time.time()
        data_rdd = data_rdd.map(lambda x: x[1]).persist(StorageLevel.MEMORY_AND_DISK)
        points = data_rdd.collectAsMap() # {idx: (attrs...), ...}
        centroids_list = [] # [(), ...]
        distance = {} # {idx: dis, ...} record the current distance for each node
        for i in range(size):
            if i == 0:
                random.seed(RANDOM_SEED)
                first_centroid_tag = random.choice(list(points.keys()))
                first_centroid = (first_centroid_tag, points[first_centroid_tag])
                centroids_list.append(first_centroid)
                distance = data_rdd.map(lambda x: computeDistanceForEachNode(x, first_centroid)).collectAsMap()
            else:
                largest_one = [None, -1]
                for k in distance:
                    if distance[k] > largest_one[1]:
                        largest_one = [k, distance[k]]
                next_centroid_tag = largest_one[0]
                next_centroid = (next_centroid_tag, points[next_centroid_tag])
                centroids_list.append(next_centroid)
                new_distance = data_rdd.map(lambda x: computeDistanceForEachNode(x, next_centroid)) \
                    .map(lambda x: (x[0], min(x[1], distance[x[0]]))) \
                    .collectAsMap()
                distance = new_distance
            # print('num of centroids', len(centroids_list))

        centroids = dict(centroids_list)
        print('Duration - Get initial clusters: %fs.' % (time.time() - t_1))
        return centroids
    else:
        return None

def computeDistance(v1, v2):
    # Euclidean Distance
    pingfanghe = 0
    for i in range(len(v1)):
        pingfanghe += (v1[i] - v2[i]) ** 2
    return sqrt(pingfanghe)

def clusterPoint(point, centroids):
    """
    :param - (idx, (cluster_tag, (attrs...)))
    :param - {cluster_tag: (attrs...), ...}
    """
    attrs = point[1][1]
    min_distance = (None, LARGE_NUMBER)
    for c in centroids:
        dis = computeDistance(attrs, centroids[c])
        if dis < min_distance[1]:
            min_distance = (c, dis)
    ctag = min_distance[0]
    return (point[0], (ctag, attrs))

def addList(l1, l2):
    """
    :param list
    """
    return [l1[i] + l2[i] for i in range(len(l1))]

def computeCentroids(points):
    """
    :param points: rdd - [(idx, (cluster_tag, (attrs...))), ...]
    :return dict - {cluster_tag: (coordinates...), ...}
    """
    new_centroids = points.map(lambda x: x[1]) \
        .map(lambda x: (x[0], (x[1], 1))) \
        .reduceByKey(lambda x, y: (addList(x[0], y[0]), x[1]+y[1])) \
        .mapValues(lambda v: tuple([x / v[1] for x in v[0]])) \
        .collectAsMap()
    return new_centroids

def checkCentroidsChanged(new_centroids, old_centroids, equal_threshold=0.0001):
    """
    :param new_centroids, old_centroids: dict - {cluster_tag: (coordinates...), ...}
    """
    if old_centroids == None:
        return True
    if len(old_centroids) != len(new_centroids):
        return True

    sumsq_all = 0
    for k in old_centroids:
        old_centroid = old_centroids[k]
        new_centroid = new_centroids[k]
        sumsq_ = 0
        for i in range(len(old_centroid)):
            sumsq_ += (old_centroid[i] - new_centroid[i]) ** 2
        sumsq_all += sumsq_ / len(old_centroid)
    sumsq_all_avg = sumsq_all / len(old_centroids)
    dis_all_avg = sqrt(sumsq_all_avg)
    print(dis_all_avg)
    if dis_all_avg > equal_threshold:
        return True
    else:
        return False  

def KMeansRDD(sc, data, k_clusters, max_iter=KMEANS_MAX_ITER):
    """
    :param data: [(idx, (attrs...)), ...]
    :return rdd: [(idx, (cluster_tag, (attrs...))), ...]
    """
    data_rdd = sc.parallelize(data).map(lambda x: (x[0], x)).persist(StorageLevel.MEMORY_AND_DISK)
    # data_rdd: [(idx, (cluster_tag, (attrs...))), ...]
    
    initial_centroids = getInitialCentroids(data_rdd, k_clusters, 'kmeans++')
    old_centroids = None
    new_centroids = initial_centroids
    
    go_on = True
    num_iter = 0
    while(go_on):
        num_iter += 1
        points_tags = data_rdd.map(lambda x: clusterPoint(x, new_centroids))

        old_centroids = new_centroids
        new_centroids = computeCentroids(points_tags)

        if num_iter >= max_iter:
            print("finish %d iterations" % (num_iter))
            break

        go_on = checkCentroidsChanged(new_centroids, old_centroids, equal_threshold=EQUAL_THRESHOLD)
        print("finish %d iterations" % (num_iter))

    
    print('kmeans totally %d iterations.' % (num_iter))
    
    return points_tags

def simpleStatistics(X, pre_state=None):
    """
    :param pre_state: list - [0, [0]*d, [0]*d] [N, SUM_vector, SUMSQ_vector]
    """
    X = list(X)
    d = len(X[0]) # dimension
    
    state = None
    if pre_state == None:
        state = [0, [0]*d, [0]*d]
    else:
        state = pre_state
        
    for x in X:
        state[0] += 1
        for i in range(d):
            state[1][i] += x[i]
            state[2][i] += x[i] ** 2
    return state

def getDiscardSets(points):
    """
    :param points: rdd - [(idx, (cluster_tag, (attrs...))), ...]
    :return : dict - {tag: [N, SUM_vector, SUMSQ_vector], ...}
    """
    initial_DS = points.map(lambda x: x[1]) \
        .groupByKey() \
        .mapValues(lambda X: simpleStatistics(X)) \
        .collectAsMap()
    return initial_DS

def rearrangeResult(res_rdd, RS_idx=None):
    """
    :param res_rdd: rdd - [(idx, (cluster_tag, (attrs...))), ...]
    :return : list - [(idx, tag), ...] idx: int tag: int
    """
    if RS_idx == None:
        return res_rdd.map(lambda x: (x[0], x[1][0])).collect()
    else:
        return res_rdd.filter(lambda x: x[0] not in RS_idx).map(lambda x: (x[0], x[1][0])).collect()

def shouldGetDSAgain(DS):
    # len(DS) < n_cluster
    if len(DS) < n_cluster:
        print('bad initial DS because the number of clusters is less than n_cluster')
        return True
    # any value of DS has only one element
    for tag in DS:
        # N == 1
        if DS[tag][0] == 1:
            print('bad initial DS because include outliars')
            return True
    print("good initial DS.")
    return False

def getCSandRS(points):
    """
    :param points: rdd - [(idx, (cluster_tag, (attrs...))), ...]
    :return RS_idx: list - [idx, ...]
    :return initial_CS: dict - {tag: [N, SUM_vector, SUMSQ_vector], ...}
    :return initial_RS: list - [(idx, (attrs...)), ...]
    """
    RS_idx = points.map(lambda x: (x[1][0], x[0])) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .filter(lambda x: len(x[1]) == 1) \
        .flatMap(lambda x: x[1]) \
        .collect()
    
    initial_CS = points.filter(lambda x: x[0] not in RS_idx) \
        .map(lambda x: x[1]) \
        .groupByKey() \
        .mapValues(lambda X: simpleStatistics(X)) \
        .collectAsMap()

    initial_RS = points.filter(lambda x: x[0] in RS_idx) \
        .map(lambda x: (x[0], x[1][1])) \
        .collect()
    return RS_idx, initial_CS, initial_RS

def calculateCentroidAndDeviation(cluster):
    """
    :param cluster : list - [N, SUM_vector, SUMSQ_vector] could be any cluster in DS or CS
    :return : list - [avg_vector, sd_vector]
    """
    n = cluster[0]
    su_v = cluster[1]
    sumsq_v = cluster[2]
    
    avg_v = []
    sd_v = []
    for i in range(len(su_v)):
        su = su_v[i]
        sumsq = sumsq_v[i]
        
        avg = su / n
        sd = sqrt((sumsq / n) - (su / n) ** 2)
        
        avg_v.append(avg)
        sd_v.append(sd)
    return [avg_v, sd_v]

def mahalanobisDistancePointCluster(p_attrs, cluster_stt):
    avg_v = cluster_stt[0]
    sd_v = cluster_stt[1]
    d = len(p_attrs)
    vsq = 0
    for i in range(d):
        if sd_v[i] != 0:
            vsq += ((p_attrs[i] - avg_v[i]) / sd_v[i]) ** 2
        else:
            vsq += (p_attrs[i] - avg_v[i]) ** 2
    return sqrt(vsq)

def checkDS(point, DS, md_threshold):
    """
    :param point: tuple - (idx, (('RS', cluster_tag), (attrs...))) cluster_tag==None if no cluster
    """
    p_idx = point[0]
    p_cluster_tag = point[1][0][1]
    p_attrs = point[1][1]
    
    closest_one = [None, LARGE_NUMBER]
    for ctag in DS:
        avg_sd = calculateCentroidAndDeviation(DS[ctag])
        md = mahalanobisDistancePointCluster(p_attrs, avg_sd)
        if md < closest_one[1]:
            closest_one = [ctag, md]
    d_tag = closest_one[0]
    d_md = closest_one[1]
    if d_md < md_threshold:
        return (p_idx, (('DS', d_tag), p_attrs))
    else:
        return point

def checkCS(point, CS, md_threshold):
    """
    :param point: tuple - (idx, (('RS' or 'DS', cluster_tag), (attrs...))) cluster_tag==None if no cluster
    """
    p_idx = point[0]
    p_cluster_tag = point[1][0][1]
    p_attrs = point[1][1]
    if p_cluster_tag != None:
        # p is in DS
        return point
    
    closest_one = [None, LARGE_NUMBER]
    for ctag in CS:
        avg_sd = calculateCentroidAndDeviation(CS[ctag])
        md = mahalanobisDistancePointCluster(p_attrs, avg_sd)
        if md < closest_one[1]:
            closest_one = [ctag, md]
    c_tag = closest_one[0]
    c_md = closest_one[1]
    if c_md < md_threshold:
        return (p_idx, (('CS', c_tag), p_attrs))
    else:
        return point

def mergePoints(sc, points, k_clusters):
    """
    by kmeans
    :param points: list - [(idx, (attrs...)), ...]
    :return new_CS, new_RS, new_res_CS: dict, list, list
    """
    rest_rdd = KMeansRDD(sc, points, k_clusters).persist(StorageLevel.MEMORY_AND_DISK)
    RS_idx, new_CS, new_RS = getCSandRS(rest_rdd)
    new_res_CS = rearrangeResult(rest_rdd, RS_idx)
    return new_CS, new_RS, new_res_CS

def shouldMergeTwoClusters(sim1, sim2, threshold):
    """
    :param sim1, sim2: list - [N, SUM_vector, SUMSQ_vector]
    """
    sim_small, sim_large = None, None
    if sim1[0] > sim2[0]:
        sim_small, sim_large = sim2, sim1
    else:
        sim_small, sim_large = sim1, sim2

    point_small = tuple([sim_small[1][i] / sim_small[0] for i in range(len(sim_small[1]))])
    avg_sd = calculateCentroidAndDeviation(sim_large)
    md = mahalanobisDistancePointCluster(point_small, avg_sd)
    if md < threshold:
        return True
    else:
        return False

def computeClustersDistance(sim1, sim2):
    d = len(sim1[1])
    sumsq_ = 0
    for i in range(d):
        sumsq_ += (sim1[1][i] / sim1[0] - sim2[1][i] / sim2[0]) ** 2
    return sqrt(sumsq_)

def mergeCS(CS, res_CS, md_threshold):
    # treat the small CS as a point
    cs_tags = list(CS.keys())
    length_cs = len(CS)
    for i in range(length_cs):
        cs_tag1 = cs_tags[i]
        sim1 = CS[cs_tag1]
        for j in range(i+1, length_cs):
            cs_tag2 = cs_tags[j]
            sim2 = CS[cs_tag2]
            if shouldMergeTwoClusters(sim1, sim2, md_threshold):
                new_n = sim1[0] + sim2[0]
                new_su_v = addList(sim1[1], sim2[1])
                new_sumsq_v = addList(sim1[2], sim2[2])
                
                # update res_CS, directly modify in the outside variable
                for i, t in enumerate(res_CS):
                    if t[1] == cs_tag2:
                        res_CS[i] = (t[0], cs_tag1)
                # update CS, directly modify in the outside variable
                print('merge CS %s to CS %s.' % (cs_tag2, cs_tag1))
                CS[cs_tag1] = [new_n, new_su_v, new_sumsq_v]
                CS.pop(cs_tag2, None)
                return True
    return False

def mergeCSwithDS(CS, DS, res_CS_dict, res_DS, md_threshold):
    """
    :param res_CS_dict: dict - {tag: [idx, ...], ...}
    """
    for cs_tag in CS.keys():
        sim_cs = CS[cs_tag]
        for ds_tag in DS.keys():
            sim_ds = DS[ds_tag]
            if shouldMergeTwoClusters(sim_cs, sim_ds, md_threshold):
                print("merge CS %s with DS %s." % (cs_tag, ds_tag))
                new_n = sim_cs[0] + sim_ds[0]
                new_su_v = addList(sim_cs[1], sim_ds[1])
                new_sumsq_v = addList(sim_cs[2], sim_ds[2])

                # append new points into res_DS, directly modify in the outside variable
                for idx in res_CS_dict[cs_tag]:
                    res_DS.append((idx, ds_tag))
                    
                # remove points from res_CS_dict
                res_CS_dict.pop(cs_tag, None)
                
                # update DS, CS, directly modify in the outside variable
                DS[ds_tag] = [new_n, new_su_v, new_sumsq_v]
                CS.pop(cs_tag, None)
                return True
    return False

def mergeCSwithDS2(CS, DS, res_CS_dict, res_DS, md_threshold):
    """
    combine CS with CS, DS with DS, CS with DS
    :param res_CS_dict: dict - {tag: [idx, ...], ...}
    """
    # generate keys
    ds_keys = list(DS.keys())
    cs_keys = list(CS.keys())
    ds_keys + cs_keys

    for ds_tag in DS.keys():
        sim_ds = DS[ds_tag]
        for cs_tag in CS.keys():
            sim_cs = CS[cs_tag]
            if shouldMergeTwoClusters(sim_cs, sim_ds, md_threshold):
                print("merge CS %s with DS %s." % (cs_tag, ds_tag))
                new_n = sim_cs[0] + sim_ds[0]
                new_su_v = addList(sim_cs[1], sim_ds[1])
                new_sumsq_v = addList(sim_cs[2], sim_ds[2])

                # append new points into res_DS, directly modify in the outside variable
                for idx in res_CS_dict[cs_tag]:
                    res_DS.append((idx, ds_tag))
                    
                # remove points from res_CS_dict
                res_CS_dict.pop(cs_tag, None)
                
                # update DS, CS, directly modify in the outside variable
                DS[ds_tag] = [new_n, new_su_v, new_sumsq_v]
                CS.pop(cs_tag, None)
                return True
    return False

def runKMeansForInitialization(sc, points_rdd):
    """
    :param points_rdd: rdd - [(idx, (attrs...)), ...] idx: int attr: float
    :return DS, CS: dict - {tag: [N, SUM_vector, SUMSQ_vector], ...}
    :return RS: list - [(idx, (attrs...)), ...]
    :return res_DS, res_CS: list - [(idx, tag), ...] idx: int tag: int
    :return RS_idx: list - [idx, ...]
    :return d: int - number of dimensions of data
    :return md_threshold: int - the threshold of Mahalanobis Distance
    """
    data_points = points_rdd.collect()

    # get number of dimensions
    d = len(data_points[0][1])
    md_threshold = sqrt(d) * ALPHA_FOR_MD

    random_seed = RANDOM_SEED
    bad_DS = True
    while(bad_DS):
        sample, rest_data = splitData(data_points, SMALL_SAMPLE_RATIO, random_seed)

        k_clusters = n_cluster
        sample_rdd = KMeansRDD(sc, sample, k_clusters).persist(StorageLevel.MEMORY_AND_DISK)
        # sample_rdd rdd: [(idx, (cluster_tag, (attrs...))), ...]
        DS = getDiscardSets(sample_rdd)
        res_DS = rearrangeResult(sample_rdd, None)
        bad_DS = shouldGetDSAgain(DS)
        random_seed += 1
    print('\n')

    k_clusters = n_cluster * MULTI_FOR_FIRST_ROUND
    rest_rdd = KMeansRDD(sc, rest_data, k_clusters).persist(StorageLevel.MEMORY_AND_DISK)
    RS_idx, CS, RS = getCSandRS(rest_rdd)
    res_CS = rearrangeResult(rest_rdd, RS_idx)
    print('\n')

    return DS, CS, RS, res_DS, res_CS, d, md_threshold

def runBFR(sc, points_rdd, res):
    DS, CS, RS, res_DS, res_CS, d, md_threshold = res

    dcr = points_rdd.map(lambda x: (x[0],(('RS', None), x[1]))) \
        .map(lambda x: checkDS(x, DS, md_threshold)) \
        .map(lambda x: checkCS(x, CS, md_threshold)) \
        .collect()
    # list - [(idx, (('DS' or 'CS' or 'RS', cluster_tag), (attrs...))), ...] cluster_tag==None if in 'RS'

    for point in dcr:
        p_idx = point[0]
        p_in = point[1][0][0]
        p_ctag = point[1][0][1]
        p_attrs = point[1][1]
        
        if p_in == 'DS':
            # DS
            sim_stt = DS[p_ctag]
            n = sim_stt[0] + 1
            su = [sim_stt[1][i] + p_attrs[i] for i in range(d)]
            sumsq = [sim_stt[2][i] + p_attrs[i] ** 2 for i in range(d)]
            DS[p_ctag] = [n, su, sumsq]
            
            # res_DS
            res_DS.append((p_idx, p_ctag))
        elif p_in == 'CS':
            # CS
            sim_stt = CS[p_ctag]
            n = sim_stt[0] + 1
            su = [sim_stt[1][i] + p_attrs[i] for i in range(d)]
            sumsq = [sim_stt[2][i] + p_attrs[i] ** 2 for i in range(d)]
            CS[p_ctag] = [n, su, sumsq]
            
            # res_CS
            res_CS.append((p_idx, p_ctag))
        else:
            # RS
            RS.append((p_idx, p_attrs))

    k_clusters = n_cluster * MULTI_FOR_OTHER_ROUND
    if len(RS) > k_clusters:
        print('RS length before kmeans:', len(RS))
        new_CS, new_RS, new_res_CS = mergePoints(sc, RS, k_clusters)
        # update
        for tag in new_CS:
            CS[tag] = new_CS[tag]
        res_CS.extend(new_res_CS)
        RS = new_RS
        print('RS length after kmeans:', len(RS))
        print('\n')

    # Merge clusters in CS that have a Mahalanobis Distance < 𝛼√𝑑.
    continue_merge = True
    while(continue_merge):
        continue_merge = mergeCS(CS, res_CS, md_threshold)

    return DS, CS, RS, res_DS, res_CS, d, md_threshold

def afterLastLoad(res):
    DS, CS, RS, res_DS, res_CS, d, md_threshold = res
    
    res_CS_dict = {}
    for x in res_CS:
        if res_CS_dict.get(x[1]) == None:
            res_CS_dict[x[1]] = [x[0]]
        else:
            res_CS_dict[x[1]].append(x[0])
    
    while(1):
        if mergeCSwithDS(CS, DS, res_CS_dict, res_DS, md_threshold) or mergeCS(CS, res_CS, md_threshold):
            continue
        else:
            break

    if res_CS_dict == {}:
        res_CS = []
        print('clear CS')
    else:
        res_CS = [(idx, tag) for tag in res_CS_dict for idx in res_CS_dict[tag]]
        print('not clear CS, having %d outliers in CS.' % (len(res_CS)))
    
    return DS, CS, RS, res_DS, res_CS, d, md_threshold

def integrateResult(res):
    DS, CS, RS, res_DS, res_CS, d, md_threshold = res
    # RS: list - [(idx, (attrs...)), ...]
    res_CS = [(x[0], -1) for x in res_CS]
    res_RS = [(x[0], -1) for x in RS]
    res_final = res_DS + res_CS + res_RS
    return res_final

def outputForFile1(res_list):
    # res_list: list - [(idx, cluster_tag), ...] idx, cluster_tag are int
    o1 = [(str(x[0]), x[1]) for x in sorted(res_list, key=lambda x: x[0])]
    o1 = dict(o1)
    with open(out_file1, 'w', encoding='utf-8') as fp:
        json.dump(o1, fp)

def getStatisticsForEachRound(n_round, res):
    DS, CS, RS, res_DS, res_CS, d, md_threshold = res
    round_id = n_round
    nof_cluster_discard = len(DS)
    nof_point_discard = len(res_DS)
    nof_cluster_compression = len(CS)
    nof_point_compression = len(res_CS)
    nof_point_retained = len(RS)
    return (round_id, nof_cluster_discard, nof_point_discard, nof_cluster_compression, nof_point_compression, nof_point_retained)

def outputForFile2(stat):
    # stat: list - [(6 int), ...]
    header = "round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained"
    with open(out_file2, 'w', encoding='utf-8') as fp:
        fp.write(header)
        fp.write('\n')
        for s in stat:
            fp.write("%d,%d,%d,%d,%d,%d\n" % (s[0], s[1], s[2], s[3], s[4], s[5]))


def main():
    t_start = time.time()
    sc = getSparkEnv()
    res = None
    stat_file2 = []

    files_complete_path = getFilesPath(input_path)

    files_length = len(files_complete_path)
    for i in range(files_length):
        file_path = files_complete_path[i]
        n_round = i + 1
        data_points_rdd = loadFileRDD(sc, file_path)
        if n_round == 1:
            res = runKMeansForInitialization(sc, data_points_rdd)
            stat_file2.append(getStatisticsForEachRound(n_round, res))
        elif 1 < n_round and n_round < files_length:
            res = runBFR(sc, data_points_rdd, res)
            stat_file2.append(getStatisticsForEachRound(n_round, res))
        else:
            # last round
            res = runBFR(sc, data_points_rdd, res)
            res = afterLastLoad(res)
            stat_file2.append(getStatisticsForEachRound(n_round, res))
            res_DCR = integrateResult(res)

    outputForFile1(res_DCR)
    outputForFile2(stat_file2)
    duration = time.time() - t_start
    print('Duration: %fs' % (duration))
    return duration

if __name__ == "__main__":
    main()

