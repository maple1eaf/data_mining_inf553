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
from task2support import *

# for run on vocareum
# import os
# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

DECIMAL_ACCURACY = 0.00000000001

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]

def buildGraph(vtx_names, edges):
    g = UndirectedGraph()
    # add vertices
    for v in vtx_names:
        g.addVertex(v)
    # add edges
    for e in edges:
        g.addEdge(e[0], e[1], e[2])
    return g

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

def rearrageCommunity(res):
    for item in res:
        item.sort()
    res.sort(key=lambda x: x[0])
    res.sort(key=lambda x: len(x))
    return res

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
    g = buildGraph(vertices_have_link, edges_list)
    
    # count betweenness
    betweenness = g.getBetweenness()
    return betweenness, vertices_have_link, edges_list, uid_map_table

def findCommunitiesWithHighestModularity(vtx_names, edges, uid_map_table):
    original_g = buildGraph(vtx_names, edges)
    g = buildGraph(vtx_names, edges)

    # count modularity, count betweenness, remove edges
    num_remain_edges = g.num_edge
    res = []
    while(num_remain_edges != 0):
        modularity = g.countModularityDependOnOriginalGraph(original_g)
        g.getBetweenness()
        removed_edges = g.removeEdgesWithHighestBetweenness()
        num_remain_edges -= len(removed_edges)
        res.append((modularity, removed_edges))

    # find highest modularity
    largest_i = None
    largest_item = res[0]
    for i, item in enumerate(res):
        if item[0] - largest_item[0] > DECIMAL_ACCURACY:
            largest_item = item
            largest_i = i
    
    # print(largest_item)

    # for i, item in enumerate(res):
    #     print(i, item)

    # gather edges which shuold be removed
    edges_should_be_removed = [i for x in res[:largest_i] for i in x[1]]

    # print(len(edges_should_be_removed))

    # rebuild the graph and remove edges to the state with highes modularity
    g = buildGraph(vtx_names, edges)
    for e in edges_should_be_removed:
        g.removeEdge(e)
    
    # get the communities and output
    res = g.getConnectedGraphs()

    for group in res:
        for i in range(len(group)):
            group[i] = uid_map_table[group[i]]
    res = rearrageCommunity(res)
    # print(len(res))
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