"""
Python: $ spark-submit task2.py <review_file> <business_file > <output_file> <if_spark> <n>
spark:
spark-submit task2.py "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/business.json" "output_task2.json" "spark" 5
no_spark:
spark-submit task2.py "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/business.json" "output_task2.json" "no_spark" 5

review _file – the input file (the review dataset) 
business_file – the input file (the business dataset) 
output_file – the output file contains your answers 
if_spark – either “spark” or “no_spark” 
n – top n categories with highest average stars (see 4.2.1)
"""
import json
import time
import sys

review_file = sys.argv[1]
business_file = sys.argv[2]
output_file = sys.argv[3]
if_spark = sys.argv[4]
n = int(sys.argv[5])

def timer(func):
    """timer decorator"""
    def wrapper(*arg, **karg):
        t1 = time.time()
        res = func(*arg, **karg)
        t2 = time.time()
        delta_t = t2 - t1
        print('time consume:%.2fs' % delta_t)
        return res
    return wrapper

def write_output_file(result):
    # write on local disk instead of hdfs
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(result, fp)

@timer
def noSpark():
    from functools import reduce

    def openDataFile(file_addr, a, b):
        with open(file_addr, 'r', encoding='utf-8') as fp:
            data = []
            for line in fp.readlines():
                d_= json.loads(line)
                data.append({a: d_[a], b: d_[b]})
        return(data)

    review_data = openDataFile(review_file, 'business_id', 'stars')
    sub_review_data = [[x['business_id'], [x['stars']]] for x in review_data] # [[business_id, [starts]]...]
    sub_review_data.sort(key=lambda x: x[0])
    def groupBy(r_l):
        li = [r_l[0]]
        for i in range(1, len(r_l)):
            key_ = r_l[i][0]
            value_ = r_l[i][1]
            if key_ == r_l[i-1][0]:
                li[-1][1] += value_
            else:
                li.append(r_l[i])
        return li
    grouped_sub_review_data = groupBy(sub_review_data)
    sub_review_data_combine_list = [[x[0], [sum(x[1]), len(x[1])]] for x in grouped_sub_review_data]
    sub_review_data_combine_dict = {}
    for x in sub_review_data_combine_list:
        sub_review_data_combine_dict[x[0]] = x[1]

    business_data = openDataFile(business_file, 'business_id', 'categories')
    filtered_business_data = [x for x in business_data if x['categories'] != None]
    sub_business_data = [[x['business_id'], [x.strip() for x in x['categories'].split(',')]] for x in filtered_business_data]
    sub_business_data_flat = [[x, item[0]] for item in sub_business_data for x in item[1]] # [[category, business_id], ...]
    sub_business_data_flat.sort(key=lambda x: x[0])
 
    def joinBR(bl, rd):
        """
        bl [[category, business_id], ...]
        rd {business_id: [sum, count], ...}
        """
        l_ = []
        for i in range(len(bl)):
            key_ = bl[i][0]
            value_ = rd.get(bl[i][1])
            if value_ == None:
                continue
            else:
                l_.append([key_, value_])
        return l_
    
    joined_b_r = joinBR(sub_business_data_flat, sub_review_data_combine_dict) # [[category, [sum_stars, 5]], ...]
    def groupByJoinedBR(jl):
        """
        jl: [[category, [sum_stars, 5]], ...]
        """
        jl1 = [[x[0], [x[1]]] for x in jl]
        jl1.sort(key=lambda x: x[0])
        li = [jl1[0]]
        for i in range(1, len(jl1)):
            k_ = jl1[i][0]
            v_ = jl1[i][1]
            if k_ == jl1[i-1][0]:
                li[-1][1] += v_
            else:
                li.append(jl1[i])
        return li
    groupedbr = groupByJoinedBR(joined_b_r)

    def reducefunc(scl):
        x_0 = scl[0][0]
        x_1 = scl[0][1]
        for j in range(1, len(scl)):
            v_0 = scl[j][0]
            v_1 = scl[j][1]
            
            x_0 += v_0
            x_1 += v_1
        result = x_0/x_1
        return result

    for x in groupedbr:
        x[1] = reducefunc(x[1])

    result_sort = groupedbr
    result_sort.sort(key=lambda x: x[1], reverse=True)
    result = {"result": result_sort[:n]}

    write_output_file(result)

@timer
def spark():
    from pyspark import SparkContext

    # define spark env
    sc = SparkContext(master="local[*]", appName="task2")

    business_data = sc.textFile(business_file).cache()
    sub_business_data = business_data.map(json.loads) \
        .filter(lambda b: b['categories'] != None) \
        .map(lambda b: (b['business_id'], [x.strip() for x in b['categories'].split(',')])) \
        .flatMap(lambda item: [(item[0], x) for x in item[1]])
    # print(sub_business_data)

    review_data = sc.textFile(review_file).cache()
    sub_review_data = review_data.map(json.loads) \
        .map(lambda r: (r['business_id'], (r['stars'], 1))) \
        .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1]))
    # print(sub_review_data)

    data_join_on_bussiness_id = sub_business_data.join(sub_review_data) \
        .map(lambda x: x[1]) \
        .reduceByKey(lambda x, y: [x[0]+y[0], x[1]+y[1]]) \
        .mapValues(lambda l: l[0]/l[1]) \
        .sortBy(keyfunc=lambda x: x[0], ascending=True) \
        .sortBy(keyfunc=lambda x: x[1], ascending=False) \
        .map(lambda x: [x[0], x[1]]) \
        .take(n)
    # print(data_join_on_bussiness_id)

    result = {'result': data_join_on_bussiness_id}
    write_output_file(result)

if __name__ == "__main__":
    if if_spark == "spark":
        spark()
    elif if_spark == "no_spark":
        noSpark()
    else:
        print('wrong if_spark value.')

