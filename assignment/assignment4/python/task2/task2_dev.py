"""
spark-submit task2.py <filter_threshold> <input_file_path> <betweenness_output_file_path> <community_output_file_path>

spark-submit task2_dev.py 7 "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw4/data/ub_sample_data.csv" "./output/betweenness_dev.txt" "./output/community_dev.txt"

vocareum:
spark-submit task2.py 7 $ASNLIB/publicdata/ub_sample_data.csv task2_betweenness.txt task2_community.txt
"""

import sys
import time
import json
import os

from pyspark import SparkConf, SparkContext, StorageLevel
import task2support

# for run on vocareum
# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]

def outputBetweennessAsFile(betweenness, uid_map_table):
    # rearrange
    b = []
    for pair in betweenness:
        b.append((tuple(sorted([uid_map_table[u] for u in pair])), betweenness[pair]))
    b.sort(key=lambda x: x[0][0])
    b.sort(key=lambda x: x[1], reverse=True)
    with open(betweenness_output_file_path, 'w', encoding='utf-8') as fp:
        for t in b:
            fp.write('%s, %s\n' % (t[0], t[1]))

def rearrageCommunities(communities):
    for item in communities:
        item.sort()
    communities.sort(key=lambda x: x[0])
    communities.sort(key=lambda x: len(x))
    return communities

def outputAsFile(res):
    with open(community_output_file_path, 'w', encoding='utf-8') as fp:
        for t in res:
            if len(t) == 1:
                fp.write("'%s'\n" % (t[0]))
            else:
                fp.write("'" + ("', '").join(t) + "'\n")

def countBetweennessByGirvanNewmanAlgorithm(sc):
    data_with_header = sc.textFile(input_file_path)
    header = data_with_header.first()
    distinct_data = data_with_header.filter(lambda l: l != header) \
        .map(lambda l: tuple(l.split(','))) \
        .distinct()\
        .persist(StorageLevel.MEMORY_AND_DISK)
    # distinct_data: RDD - [(uid, bid), ...]

    uid_map_table = distinct_data.map(lambda x: x[0]).distinct().collect()
    bid_map_table = distinct_data.map(lambda x: x[1]).distinct().collect()

    baskets = distinct_data.map(lambda x: (uid_map_table.index(x[0]), bid_map_table.index(x[1])))\
        .map(lambda x: (x[0], [x[1]])) \
        .reduceByKey(lambda x, y: x + y) \
        .collectAsMap()

    uids = sorted(list(baskets.keys()))
    uids_length = len(uids)

    edges_list = []
    vertices_have_link = set([])
    for i in range(uids_length):
        uid1 = uids[i]
        set1 = set(baskets[uid1])
        for j in range(i+1, uids_length):
            uid2 = uids[j]
            set2 = set(baskets[uid2])
            if len(set1.intersection(set2)) >= filter_threshold:
                edges_list.append((uid1, uid2, 1))
                vertices_have_link.add(uid1)
                vertices_have_link.add(uid2)

    # build graph
    g = task2support.UndirectedGraph(vertices=vertices_have_link, edges=edges_list)
    
    # count betweenness
    betweenness = g.getBetweenness()
    return betweenness, vertices_have_link, edges_list, uid_map_table

def findCommunitiesWithHighestModularity(vtx_names, edges, uid_map_table):
    g = task2support.UndirectedGraph(vertices=vtx_names, edges=edges)

    optimal_groups = g.getOptimalClustersBasedOnModularity()

    for group in optimal_groups:
        for i in range(len(group)):
            group[i] = uid_map_table[group[i]]

    res = rearrageCommunities(optimal_groups)

    return res

if __name__ == "__main__":
    t_start = time.time()

    conf = SparkConf() \
        .setAppName("task1") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    betweenness, vertices, edges, uid_map_table= countBetweennessByGirvanNewmanAlgorithm(sc)

    outputBetweennessAsFile(betweenness, uid_map_table)

    communities = findCommunitiesWithHighestModularity(vertices, edges, uid_map_table)

    outputAsFile(communities)

    t_end = time.time()
    print('time: %fs' % (t_end-t_start))