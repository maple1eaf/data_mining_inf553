"""
spark-submit firstname_lastname_task2.py <filter threshold> <support> <input_file_path> <output_file_path>

spark-submit task2_develop.py 70 50 "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/task2_data.csv" "task2_develop_output.txt"
"""
import sys
import time
from itertools import combinations

from pyspark import SparkConf, SparkContext
import psutil
import os

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

info = psutil.virtual_memory()

memory_1 = psutil.Process(os.getpid()).memory_info().rss

# caution: filter_threshold, support are str
filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

def timer(beginstr='', endstr='Time Consuming: '):
    """timer decorator"""
    def decor(func):
        def wrapper(*arg, **karg):
            t1 = time.time()
            if beginstr != '':
                print(beginstr)
            res = func(*arg, **karg)
            t2 = time.time()
            delta_t = t2 - t1
            print('%s%.3fs' % (endstr, delta_t))
            return res
        return wrapper
    return decor

def writeIntoFile(filepath, candidates, frequent_itemsets):
    """
    either candidates or frequent_itemsets is a dict which is the output of function groupAndSort
    ex: {1:[a,b,c], 2:[(a,b), (c,d)], ...}
    """
    with open(filepath, 'w', encoding='utf-8') as fp:
        def writing(title, content):
            fp.write(title + "\n")
            if content == {}:
                fp.write("\n")
            else:
                for key in sorted(content.keys()):
                    if key == 1:
                        words = "('" + ("'),('").join(content[key]) + "')\n\n"
                        fp.write(words)
                    else:
                        words = (",").join([str(x) for x in content[key]]) + "\n\n"
                        fp.write(words)
        writing("Candidates:", candidates)
        writing("Frequent Itemsets:", frequent_itemsets)

class Bitmap():
    def __init__(self, list_size):
        """
        list_size - size of a list. the list consists of either 1 or 0
        """
        self.size = list_size // 31 + 1
        self.array = [0] * self.size
    
    def getValue(self, n):
        """
        n - position number, from 0
        """
        ai, bp = self.getPosition(n)
        return self.array[ai] >> bp & 1

    def setValue(self, n, v):
        """
        n - position number, from 0
        v - value, 1 or 0
        """
        ai, bp = self.getPosition(n)
        if v == 1:
            # set the bit to one
            self.array[ai] = self.array[ai] | (1 << bp)
        elif v == 0:
            # set the bit to zero
            self.array[ai] = self.array[ai] & (~(1 << bp))
        else:
            print("wrong v value.")

    def getPosition(self, n):
        """
        n - position number, from 0
        """
        array_index = n // 31
        bit_position = n % 31
        return(array_index, bit_position)
    
    def initialize(self, l):
        for n, v in enumerate(l):
            self.setValue(n, v)

class ProjectionTable():
    """
    key-value domains are one to one projection.
    """
    def __init__(self, values):
        """
        values - iterable
        iv - index-value pair
        """
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

def hashPCY(x1, x2, n):
    return x1 % n

def generateSuperItemsets(base_itemsets):
    """
    combine tuples in the base itemsets list to generate the immediate super itemsets list
    input:
    base_itemsets - [(a,b), (b,c), (a,c) ...]
    output:
    super_itemsets - [(a,b,c), ...]
    """
    # way 1 faster
    t1 = time.time()
    if base_itemsets == []:
        return []

    if type(base_itemsets[0]) is int:
        # generate C2
        base_itemsets.sort()
        len_itemsets = len(base_itemsets)
        c2 = []
        for n_x in range(len_itemsets-1):
            for n_y in range(n_x+1, len_itemsets):
                xy_list = [base_itemsets[n_x], base_itemsets[n_y]]
                c2.append(tuple(xy_list))
        t2 = time.time()
        print("L1->C2 takes %fs." % (t2-t1))
        return c2
    else:
        # generate candidates with # of items more than 2
        for n in range(len(base_itemsets)):
            base_itemsets[n] = sorted(base_itemsets[n])

        num_base = len(base_itemsets[0])
        num_super = num_base + 1

        super_itemsets = []
        len_itemsets = len(base_itemsets)
        for n_x in range(len_itemsets):
            x = base_itemsets[n_x]
            for n_y in range(n_x+1, len_itemsets):
                y = base_itemsets[n_y]
                if x[:-1] == y[:-1] and x[-1] < y[-1]:
                    xy_list = x + y[-1:]
                    count_ = 0
                    for i in range(len(xy_list)):
                        if xy_list[:i]+xy_list[i+1:] in base_itemsets:
                            count_ += 1
                        else:
                            break
                    if count_ == num_super:
                        super_itemsets.append(tuple(xy_list))
        t2 = time.time()
        print("L%d->C%d takes %fs." % (num_super-1, num_super, t2-t1))
        return super_itemsets


    # # way 2 slower
    # t1 = time.time()
    # if base_itemsets == []:
    #     return []
    # num_base = len(base_itemsets[0])
    # num_super = num_base + 1
    # # use iterator here instead of list can save massive data
    # combines = combinations(base_itemsets, num_super)
    # super_itemsets = []
    # while(1):
    #     try:
    #         c = next(combines)
    #         x = set([j for i in c for j in i])
    #         if len(x) == num_super:
    #             super_itemsets.append(tuple(sorted(list(x))))
    #     except StopIteration:
    #         break
    # t2 = time.time()
    # print("generate sets contain %d items consume %fs." % (num_super, t2-t1))
    # print("%d super itemsets:" % (num_super), super_itemsets)
    # return super_itemsets


def PCY(baskets, support_threshold, n_buckets=1000):
    """
    input:
        baskets - [[a,b,c], ...] a,b,c are integers
        support_threshold - support threshold
        n_buckets - number of buckets
    output:
        frequent_itemsets - a list of frequent itemsets (integers), [12, 5, 6, (12, 5), ...]
    """
    # 1st pass
    frequent_itemsets = []

    baskets = list(baskets)
    # print('---baskets[0]=', baskets[0])
    single_item_count = {}
    buckets_list = [0] * n_buckets
    print('---initial buckets_list size:---', len(buckets_list))
    # number_baskets = 0
    for basket in baskets:
        # print("---process %d basket in a partition---" % number_baskets)
        # number_baskets +=1
        basket.sort()
        n_items = len(basket)
        for i, item in enumerate(basket):
            # count single items
            if item not in single_item_count.keys():
                single_item_count[item] = 1
            else:
                single_item_count[item] += 1
            # hash pairs
            if i < n_items-1:
                for j in range(i+1, n_items):
                    bucket_idx = hashPCY(basket[i], basket[j], n_buckets)
                    buckets_list[bucket_idx] += 1
    print("---pcy_1.1stage---")
    # print("---buckets_list---:", buckets_list)
    # check frequent single item
    fs_list = [] # frequent single item list
    for item in single_item_count.keys():
        count = single_item_count[item]
        if count >= support_threshold:
            # fs_list.append([item, count])
            fs_list.append(item)
    print("---frequent single item---:", fs_list[:10])
    frequent_itemsets.extend(fs_list)

    # create frequent buckets bitmap
    for i in range(len(buckets_list)):
        # print(i)
        if buckets_list[i] >= support_threshold:
            buckets_list[i] = 1
        else:
            buckets_list[i] = 0
    bitmap = Bitmap(len(buckets_list))
    print('+++len(buckets_list):+++', len(buckets_list))
    bitmap.initialize(buckets_list)
    print('+++bitmap.size is:+++', bitmap.size)
    print('+++bitmap is:+++', bitmap.array)

    # 2nd pass
    # print('pass222222222222', baskets[0])
    count_pairs = {}
    for basket in baskets:
        # notice that every basket in baskets are already sorted
        # basket.sort()
        n_items = len(basket)
        for i, item_i in enumerate(basket):
            # hash pairs
            if item_i in fs_list and i < n_items-1:
                for j in range(i+1, n_items):
                    item_j = basket[j]
                    if item_j in fs_list:
                        ij_bitmap_position = hashPCY(item_i, item_j, n_buckets)
                        if bitmap.getValue(ij_bitmap_position) == 1:
                            pair = (item_i, item_j)
                            if pair in count_pairs.keys():
                                count_pairs[pair] += 1
                            else:
                                count_pairs[pair] = 1
    fp_list = [] # frequent pairs list
    for pair in count_pairs.keys():
        count = count_pairs[pair]
        if count >= support_threshold:
            # print('frequent candidate pairs:', pair, count)
            # pair_str_for_each = (pair[0], pair[1])
            fp_list.append(pair)
    print("---frequent pairs---:", fp_list[:10])
    frequent_itemsets.extend(fp_list)

    # further passes
    base_itemsets = fp_list
    n_itemsets = 3
    while(1):
        print("-----now we're going to generate itemsets containing %d items.-----" % (n_itemsets))
        n_itemsets += 1
        super_itemsets = generateSuperItemsets(base_itemsets)
        print('---have generated itemsets of number %d---' % (n_itemsets-1))
        if super_itemsets == []:
            break
        else:
            count_further = {}
            for candidate in super_itemsets:
                count_further[candidate] = 0

            for basket in baskets:
                for candidate in super_itemsets:
                    count = 0
                    for x in basket:
                        if x in candidate:
                            count += 1
                    if count == len(candidate):
                        count_further[candidate] += 1
            # print("---count further---:", count_further)

            fi_list = [itemset for itemset in count_further.keys() if count_further[itemset] >= support_threshold]
            
            frequent_itemsets.extend(fi_list)
            base_itemsets = fi_list

    # print("---frequent itemsets in a partition---:", frequent_itemsets)
    return frequent_itemsets

def Apriori(baskets, support_threshold):
    """
    input:
        baskets - [[a,b,c], ...] a,b,c are integers
        support_threshold - support threshold
    output:
        frequent_itemsets - a list of frequent itemsets (integers), [12, 5, 6, (12, 5), ...]
    """
    # 1st pass
    frequent_itemsets = []

    baskets = list(baskets)
    # print('---baskets[0]=', baskets[0])
    single_item_count = {}
    # number_baskets = 0
    for basket in baskets:
        # print("---process %d basket in a partition---" % number_baskets)
        # number_baskets +=1
        # basket.sort()
        n_items = len(basket)
        for i, item in enumerate(basket):
            # count single items
            if single_item_count.get(item) is None:
                single_item_count[item] = 1
            else:
                single_item_count[item] += 1
    print("---apriori: count C1 done---")
    # print("---buckets_list---:", buckets_list)
    # check frequent single item
    fs_list = [item for item in single_item_count.keys() if single_item_count[item] >= support_threshold]
    frequent_itemsets.extend(fs_list)
    print("---apriori: L1 done.---", fs_list[:10])
    new_baskets = [[item for item in row if item in fs_list] for row in baskets]

    # 2nd pass
    # print('pass222222222222', baskets[0])
    t_l1 = time.time()
    count_pairs = {}
    for basket in new_baskets:
        # notice that every basket in baskets are not sorted
        basket.sort()
        n_items = len(basket)
        for i in range(n_items-1):
            for j in range(i+1, n_items):
                pair = (basket[i], basket[j])
                if count_pairs.get(pair) is None:
                    count_pairs[pair] = 1
                else:
                    count_pairs[pair] += 1

        # for i, item_i in enumerate(basket):
        #     # hash pairs
        #     if i < n_items-1:
        #         for j in range(i+1, n_items):
        #             item_j = basket[j]
        #             pair = (item_i, item_j)
        #             if count_pairs.get(pair) is None:
        #                 count_pairs[pair] = 1
        #             else:
        #                 count_pairs[pair] += 1
    fp_list = [pair for pair in count_pairs.keys() if count_pairs[pair] >= support_threshold] # frequent pairs list
    frequent_itemsets.extend(fp_list)
    print("---apriori: L2 done.", fp_list[:10])
    t_l2 = time.time()
    print('L1->L2 takes %fs' % (t_l2-t_l1))


    # further passes
    base_itemsets = fp_list
    n_itemsets = 3
    while(1):
        print("---start to generate C%d.---" % (n_itemsets))
        super_itemsets = generateSuperItemsets(base_itemsets)

        if super_itemsets == []:
            print('---C%d has no item---' % n_itemsets)
            break
        else:
            t_c = time.time()
            print('---C%d has %d items---' % (n_itemsets, len(super_itemsets)), super_itemsets[:10])
            count_further = {}

            # for basket in new_baskets:
            #     combine = combinations(basket, n_itemsets)
            #     while True:
            #         try:
            #             x = next(combine)
            #             if x in super_itemsets:
            #                 if count_further.get(x) is None:
            #                     count_further[x] = 1
            #                 else:
            #                     count_further[x] += 1
            #         except StopIteration:
                        # break
            

            for basket in new_baskets:
                basket_set = set(basket)
                for candidate in super_itemsets:
                    candidate_set = set(candidate)
                    if candidate_set.issubset(basket_set):
                        if count_further.get(candidate) is None:
                            count_further[candidate] = 1
                        else:
                            count_further[candidate] += 1
            # print("---count further---:", count_further)

            fi_list = [item for item in count_further.keys() if count_further[item] >= support_threshold]
            
            frequent_itemsets.extend(fi_list)
            print("---apriori: L%d done." % n_itemsets, fi_list[:10])
            base_itemsets = fi_list
            t_l = time.time()
            print('---C%d->L%d takes %fs---' % (n_itemsets, n_itemsets, t_l-t_c))
            n_itemsets += 1

    # print("---frequent itemsets in a partition---:", frequent_itemsets)
    return frequent_itemsets

def advanced_Apriori(baskets, support_threshold):
    """
    input:
        baskets - [[a,b,c], ...] a,b,c are integers
        support_threshold - support threshold
    output:
        frequent_itemsets - a list of frequent itemsets (integers), [12, 5, 6, (12, 5), ...]
    """
    from itertools import combinations

    t1 = time.time()
    baskets = list(baskets)

    # 1st pass
    frequent_itemsets = []

    # occur_map indicates one item appears in which baskets
    occur_map = {}
    for i_b, basket in enumerate(baskets):
        n_items = len(basket)
        for item in basket:
            if occur_map.get(item) is None:
                occur_map[item] = {i_b}
            else:
                occur_map[item].add(i_b)
    # print("---occur_map:---", occur_map)
    t2 = time.time()
    print('generate occur_map done. time:%fs' % (t2-t1))

    n_itemsets = 1
    f_set = set([])
    occur_keys = set(occur_map.keys())
    for c in combinations(occur_keys, n_itemsets):
        k = c[0]
        if len(occur_map[k]) >= support_threshold:
            f_set.add(k)
    fc_set = sorted(list(f_set))
    # print('---fs_list:---', f_set)
    # print("---occur_map after filter:---", occur_map)
    t3 = time.time()
    print('L1 done. time:%fs' % (t3-t2))
    n_itemsets += 1

    while(1):
        t4 = time.time()
        # print('n_itemsets:', n_itemsets)
        fc_set_each = set([])
        for c in combinations(fc_set, n_itemsets):
            # c is a list, like [1, 2, 3]

            set_x = occur_map[c[0]]
            for i in range(1, len(c)):
                set_y = occur_map[c[i]]
                set_x = set_x.intersection(set_y)

            if set_x != set([]) and len(set_x) >= support_threshold:
                f_set.add(tuple(c))
                fc_set_each.update(c)
        
        t5 = time.time()
        print('L%d done. time:%fs' % (n_itemsets, t5-t4))

        if len(fc_set_each) == 0:
            # print('f_set:', f_set)
            return list(f_set)
        else:
            fc_set = sorted(list(fc_set_each))
            n_itemsets += 1

@timer(endstr="second map function: time consuming:")
def secondMap(baskets, candidates):
    """
    baskets - [[a,b,c], ...]
    candidates - a list of candidates generated by the first mapreduce. a candidate is either an integer or a tuple of integers
    """
    candi_dict = {}
    for candidate in candidates:
        candi_dict[candidate] = 0


    for basket in baskets:
        basket_set = set(basket)
        for candidate in candidates:
            if type(candidate) == int:
                # itemset candidate is an integer item
                if candidate in basket:
                    candi_dict[candidate] += 1
            else:
                candidate_set = set(candidate)
                if candidate_set.issubset(basket_set):
                    candi_dict[candidate] += 1

    # for basket in baskets:
    #     for candidate in candidates:
    #         if type(candidate) == int:
    #             # itemset candidate is an integer item
    #             if candidate in basket:
    #                 candi_dict[candidate] += 1
    #         else:
    #             # itemset is a tuple of integers
    #             count = 0
    #             for x in basket:
    #                 # print(x, candidate, x in candidate)
    #                 if x in candidate:
    #                     count += 1
    #             if count == len(candidate):
    #                 candi_dict[candidate] += 1
    # # print("---count for candidates in a partition---:", candi_dict)
    candi_count_list = [(x, candi_dict[x])for x in candi_dict.keys()]
    return candi_count_list

def getBackToValues(itemset, data_pt):
    if type(itemset) == int:
        return data_pt.getValue(itemset)
    else:
        return tuple(sorted([data_pt.getValue(x) for x in itemset]))

def groupAndSort(result_list):
    """
    sort result list:
    input:
    result_list - [102, '98', ('102', '98'), ('97', '99'), ('101', '99'), '101', '97', '99', ('97', '98'), ('98', '99')]
    output:
    group = {1: ['101', '102', '97', '98', '99'], 2: [('101', '99'), ('102', '98'), ('97', '98'), ('97', '99'), ('98', '99')]}
    for example, the key 1 means the corresponding value is single item list.
    """
    group = {}
    for x in result_list:
        if type(x) == str:
            if group.get(1) == None:
                group[1] = [x]
            else:
                group[1].append(x)
        else:
            n = len(x)
            if group.get(n) == None:
                group[n] = [x]
            else:
                group[n].append(x)

    for key in group.keys():
        # print("before sort", group[key])
        group[key].sort()
        # print("after sort", group[key])

    return group

def findFrequentItemsets(data, data_pt):
    """
    data - spark pipeline, caution: now, integers in a basket instead of string
    data_pt - a projection table, an single item (string) <-> an integer
    """
    # get the number of partitions
    n_partitions = data.getNumPartitions()
    # print("number of partitions:", n_partitions)
    # decide the adjusted support threshold
    adjusted_support = int(support / n_partitions)
    if adjusted_support == 0:
        adjusted_support = 1

    # first mapreduce
    candidates = data.mapPartitions(lambda x: advanced_Apriori(x, adjusted_support)) \
        .map(lambda x: (x,1)) \
        .reduceByKey(lambda x, y: 1) \
        .map(lambda x: x[0]) \
        .collect()
    print("*** candidates in all ***:", candidates[:10])
    candidates_values_version = list([getBackToValues(x, data_pt) for x in candidates])
        
    # second mapreduce
    frequent_itemsets = data.mapPartitions(lambda x: secondMap(x, candidates)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] >= support) \
        .map(lambda x: getBackToValues(x[0], data_pt)) \
        .collect()
        
    print("*** frequent itemset on the whole ***:", frequent_itemsets[:10])
    # print('groupby:', groupAndSort(frequent_itemsets))

    # write into output file
    cvv = groupAndSort(candidates_values_version)
    fis = groupAndSort(frequent_itemsets)

    for i in fis.keys():
        print('---number of frequent itemsets containing %d items: %d---' % (i, len(fis[i])))

    writeIntoFile(output_file_path, cvv, fis)

def getData(sc, filter_threshold):
    # emit the 1st line: "user_id,business_id"
    # make baskets for user1: [business11, business12, business13, ...] ...
    data_with_header = sc.textFile(input_file_path)
    header = data_with_header.first()
    data_0 = data_with_header.filter(lambda l: l != header) \
        .map(lambda l: l.split(',')) \
        .cache()
    
    memory_2 = psutil.Process(os.getpid()).memory_info().rss
    print("memory usage:", memory_2-memory_1)

    # generate a rename table
    distinct_data = data_0.map(lambda x: x[1]).distinct().collect()
    data_pt = ProjectionTable(distinct_data)
    print(data_pt.iv[-20:])

    data = data_0.groupByKey() \
        .map(lambda x: list(set(x[1]))) \
        .filter(lambda x: len(x) > filter_threshold) \
        .map(lambda l: [data_pt.getIndex(x) for x in l]) \
        .cache()
    memory_2 = psutil.Process(os.getpid()).memory_info().rss
    print("memory usage:", memory_2-memory_1)
    
    return data, data_pt

if __name__ == "__main__":
    t_a = time.time()
    # define spark env
    conf = SparkConf() \
        .setAppName("task1") \
        .setMaster("local[*]")
    sc = SparkContext(conf=conf)

    time_before_load_file = time.time()
    print('build spark context done. time:%fs' % (time_before_load_file-t_a))
    # step 1: get data and the rename table
    data, data_pt = getData(sc, filter_threshold)
    memory_2 = psutil.Process(os.getpid()).memory_info().rss
    print("memory usage:", memory_2-memory_1)
    # print(data.take(2))
    t_b = time.time()
    print('preprocess data done. time:%fs' % (t_b-time_before_load_file))

    # step 2: run SON algorithm to find frequent itemsets
    findFrequentItemsets(data, data_pt)

    time_after_write_output_file = time.time()
    print("Duration: %d" % (time_after_write_output_file - time_before_load_file))