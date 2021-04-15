from pyspark import SparkContext, SparkConf
import json

biz_first_file = 'file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw6/asnlib/publicdata/business_first.json'
biz_second_file = 'file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw6/asnlib/publicdata/business_second.json'
my_res_file = '/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw6/python/output/task1_result'


def city_makeup(dic):
    if 'city' not in dic.keys():
        dic['city'] = ''
    return dic

def check(s, cities):
    if s == '':
        return '0'
    if s in cities:
        return '1'
    else:
        return '0'


conf = SparkConf().setMaster('local[*]')
sc = SparkContext(conf=conf)

cities = sc.textFile(biz_first_file)\
    .map(lambda s: json.loads(s))\
    .filter(lambda s: 'city' in s.keys() and len(s['city']) != 0)\
    .map(lambda s: s['city'])\
    .distinct()\
    .collect()

true_res = sc.textFile(biz_second_file)\
    .map(lambda s: json.loads(s))\
    .map(lambda s: city_makeup(s))\
    .map(lambda s: check(s['city'], cities))\
    .collect()


with open(my_res_file, 'r') as f:
    my_res = str(f.read()).split(' ')


condition_positive = 0
condition_negative = 0
false_positive = 0
false_negative = 0
for i in range(len(true_res)):
    if true_res[i] == '1':
        condition_positive += 1
    elif true_res[i] == '0':
        condition_negative += 1
    if my_res[i] == '1' and true_res[i] == '0':
        false_positive += 1
    elif my_res[i] == '0' and true_res[i] == '1':
        false_negative += 1

FPR = false_positive/condition_negative
FNR = false_negative/condition_positive
print('FPR: %f' % FPR)
print('FNR: %f' % FNR)