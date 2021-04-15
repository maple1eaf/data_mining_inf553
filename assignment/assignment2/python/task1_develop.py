"""
spark-submit firstname_lastname_task1.py <case number> <support> <input_file_path> <output_file_path>
# case 1
spark-submit task1_develop.py 1 4 "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/small1.csv" "task1_develop_output_case1.txt"
spark-submit task1_develop.py 1 10 "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/small2.csv" "task1_develop_output_case1.txt"
# case 2
spark-submit task1_develop.py 2 9 "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/small1.csv" "task1_develop_output_case2.txt"
spark-submit task1_develop.py 2 10 "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/small2.csv" "task1_develop_output_case2.txt"
"""
import sys
import time
from itertools import combinations

from pyspark import SparkConf, SparkContext

# caution: case_number, support are str
case_number = int(sys.argv[1])
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
    return (x1 * x2) % n

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
        print("generate C2 consume %fs." % (t2-t1))
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
        print("generate C%d items consume %fs." % (num_super, t2-t1))
        return super_itemsets

# def generateSuperItemsets(base_itemsets):
#     """
#     combine tuples in the base itemsets list to generate the immediate super itemsets list
#     input:
#     base_itemsets - [(a,b), (b,c), (a,c) ...]
#     output:
#     super_itemsets - [(a,b,c), ...]
#     """
#     # way 1 faster
#     t1 = time.time()
#     if base_itemsets == []:
#         return []

#     for n in range(len(base_itemsets)):
#         base_itemsets[n] = sorted(base_itemsets[n])

#     num_base = len(base_itemsets[0])
#     num_super = num_base + 1

#     super_itemsets = []
#     len_itemsets = len(base_itemsets)
#     for n_x in range(len_itemsets-1):
#         x = base_itemsets[n_x]
#         for n_y in range(n_x+1, len_itemsets):
#             y = base_itemsets[n_y]
#             xy_list = sorted(list(set(x + y)))
#             if len(xy_list) == num_super and tuple(xy_list) not in super_itemsets:
#                 count_ = 0
#                 for i in range(len(xy_list)):
#                     if xy_list[:i]+xy_list[i+1:] in base_itemsets:
#                         count_ += 1
#                     else:
#                         break
#                 if count_ == num_super:
#                     super_itemsets.append(tuple(xy_list))
#     t2 = time.time()
#     print("generate sets contain %d items consume %fs." % (num_super, t2-t1))
#     return super_itemsets


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
    for basket in baskets:
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

    # check frequent single item
    fs_list = [] # frequent single item list
    for item in single_item_count.keys():
        count = single_item_count[item]
        if count >= support_threshold:
            # fs_list.append([item, count])
            fs_list.append(item)
    # print("---frequent single item---:", fs_list)
    frequent_itemsets.extend(fs_list)

    # create frequent buckets bitmap
    for i in range(len(buckets_list)):
        # print(i)
        if buckets_list[i] >= support_threshold:
            buckets_list[i] = 1
        else:
            buckets_list[i] = 0
    bitmap = Bitmap(len(buckets_list))
    bitmap.initialize(buckets_list)
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
                    ij_bitmap_position = hashPCY(item_i, item_j, n_buckets)
                    pair = (item_i, item_j)
                    if item_j in fs_list and bitmap.getValue(ij_bitmap_position) == 1:
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
    # print("---frequent pairs---:", fp_list)
    frequent_itemsets.extend(fp_list)

    # further passes
    base_itemsets = fp_list
    n_itemsets = 3
    while(1):
        print("-----now we're going to generate itemsets containing %d items.-----" % (n_itemsets))
        n_itemsets += 1
        super_itemsets = generateSuperItemsets(base_itemsets)
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

    print("---frequent itemsets in a partition---:", frequent_itemsets[-10:])
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

    # further passes
    base_itemsets = fs_list
    n_itemsets = 2
    while(1):
        print("---start to generate C%d.---" % (n_itemsets))
        super_itemsets = generateSuperItemsets(base_itemsets)

        if super_itemsets == []:
            print('---C%d has no item---' % n_itemsets)
            break
        else:
            print('---C%d has %d items---' % (n_itemsets, len(super_itemsets)), super_itemsets[:10])
            count_further = {}
            for basket in baskets:
                for candidate in super_itemsets:
                    count = 0
                    for x in basket:
                        if x in candidate:
                            count += 1
                    if count == len(candidate):
                        if count_further.get(candidate) is None:
                            count_further[candidate] = 1
                        else:
                            count_further[candidate] += 1
            # print("---count further---:", count_further)

            fi_list = [item for item in count_further.keys() if count_further[item] >= support_threshold]
            
            frequent_itemsets.extend(fi_list)
            print("---apriori: L%d done." % n_itemsets, fi_list[:10])
            base_itemsets = fi_list
            n_itemsets += 1

    # print("---frequent itemsets in a partition---:", frequent_itemsets)
    return frequent_itemsets

def advanced_Apriori(baskets, support_threshold_all, n_all_baskets):
    """
    input:
        baskets - [[a,b,c], ...] a,b,c are integers
        support_threshold - support threshold
    output:
        frequent_itemsets - a list of frequent itemsets (integers), [12, 5, 6, (12, 5), ...]
    """
    from itertools import combinations

    baskets = list(baskets)
    # count support_threshold
    n_partition_baskets = len(baskets)
    p = n_partition_baskets / n_all_baskets
    support_threshold = p * support_threshold_all
    if support_threshold > 2:
        support_threshold = int(support_threshold)

    # 1st pass
    frequent_itemsets = []

    # print('---baskets[0]=', baskets[0])
    # occur_map indicates one item appears in which baskets
    occur_map = {}
    # number_baskets = 0
    for i_b, basket in enumerate(baskets):
        # print("---process %d basket in a partition---" % number_baskets)
        # number_baskets +=1
        # basket.sort()
        n_items = len(basket)
        for item in basket:
            # count single items
            if occur_map.get(item) is None:
                occur_map[item] = {i_b}
            else:
                occur_map[item].add(i_b)
    print("---occur_map:---", occur_map)

    n_itemsets = 1
    f_set = set([])
    occur_keys = set(occur_map.keys())
    for c in combinations(occur_keys, n_itemsets):
        k = c[0]
        if len(occur_map[k]) >= support_threshold:
            f_set.add(k)
    fc_set = f_set
    print('---fs_list:---', f_set)
    print("---occur_map after filter:---", occur_map)
    n_itemsets += 1

    while(1):
        print('n_itemsets:', n_itemsets)
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

        if len(fc_set_each) == 0:
            print('f_set:', f_set)
            return list(f_set)
        else:
            fc_set = fc_set_each
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
        for candidate in candidates:
            if type(candidate) == int:
                # itemset candidate is an integer item
                if candidate in basket:
                    candi_dict[candidate] += 1
            else:
                # itemset is a tuple of integers
                count = 0
                for x in basket:
                    # print(x, candidate, x in candidate)
                    if x in candidate:
                        count += 1
                if count == len(candidate):
                    candi_dict[candidate] += 1
    # print("---count for candidates in a partition---:", candi_dict)
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

def findFrequentItemsets(data, data_pt, n_all_baskets):
    """
    data - spark pipeline, caution: now, integers in a basket instead of string
    data_pt - a projection table, an single item (string) <-> an integer
    """
    # # get the number of partitions
    # n_partitions = data.getNumPartitions()
    # # print("number of partitions:", n_partitions)
    # # decide the adjusted support threshold
    # adjusted_support = int(support / n_partitions)
    # if adjusted_support == 0:
    #     adjusted_support = 1

    # first mapreduce
    candidates = data.mapPartitions(lambda x: advanced_Apriori(x, support, n_all_baskets)) \
        .map(lambda x: (x,1)) \
        .reduceByKey(lambda x, y: 1) \
        .map(lambda x: x[0]) \
        .collect()
    print("*** candidates ***:", candidates[-10:])
    candidates_values_version = list([getBackToValues(x, data_pt) for x in candidates])
        
    # second mapreduce
    frequent_itemsets = data.mapPartitions(lambda x: secondMap(x, candidates)) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda x: x[1] >= support) \
        .map(lambda x: getBackToValues(x[0], data_pt)) \
        .collect()
        
    print("*** frequent itemset on the whole ***:", frequent_itemsets[-10:])
    # print('groupby:', groupAndSort(frequent_itemsets))

    # write into output file
    writeIntoFile(output_file_path, groupAndSort(candidates_values_version), groupAndSort(frequent_itemsets))

def case(sc, case_number):
    # emit the 1st line: "user_id,business_id"
    # make baskets for user1: [business11, business12, business13, ...] ...
    data_with_header = sc.textFile(input_file_path)
    header = data_with_header.first()
    data_0 = ''
    if case_number == 1:
        data_0 = data_with_header.filter(lambda l: l != header) \
            .map(lambda l: l.split(',')) \
            .cache()
    elif case_number == 2:
        data_0 = data_with_header.filter(lambda l: l != header) \
            .map(lambda l: l.split(',')) \
            .map(lambda l: (l[1], l[0])) \
            .cache()
    else:
        print("wrong case_number value!")
        return None

    # generate a rename table
    distinct_data = data_0.map(lambda x: x[1]).distinct().collect()
    data_pt = ProjectionTable(distinct_data)
    print(data_pt.iv)

    data_1 = data_0.groupByKey() \
        .map(lambda x: list(set(x[1]))) \
        .cache()

    n_all_baskets = data_1.count()
    print(n_all_baskets)

    data = data_1.map(lambda l: [data_pt.getIndex(x) for x in l]) \
        .cache()

    return data, data_pt, n_all_baskets

if __name__ == "__main__":
    # define spark env
    conf = SparkConf() \
        .setAppName("task1") \
        .setMaster("local[*]")
    sc = SparkContext(conf=conf)

    time_before_load_file = time.time()

    # step 1: get data and the rename table
    data, data_pt, n_all_baskets = case(sc, case_number)

    # step 2: run SON algorithm to find frequent itemsets
    findFrequentItemsets(data, data_pt, n_all_baskets)

    time_after_write_output_file = time.time()
    print("Duration: %d" % (time_after_write_output_file - time_before_load_file))
