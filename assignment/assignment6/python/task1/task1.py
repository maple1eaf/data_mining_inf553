"""
spark-submit task1.py <first_json_path> <second_json_path> <output_file_path>

# on vocareum
spark-submit task1.py $ASNLIB/publicdata/business_first.json $ASNLIB/publicdata/business_second.json task1_ans
"""


import sys
import time
import json
import random
import binascii
# import platform

from pyspark import SparkConf, SparkContext, StorageLevel

import os
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

RANDOM_SEED = 1208
NUM_HASH = 20
BIT_ARRAY_LENGTH = 30000

# system_type = platform.system()
# if system_type == 'Linux':
#     print(system_type)
#     # for run on vocareum
#     import os
#     os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
#     os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

#     first_json_path = sys.argv[1]
#     second_json_path = sys.argv[2]
#     output_file_path = sys.argv[3]
# elif system_type == 'Darwin':
#     print(system_type)
#     # run for local macos
#     first_json_path = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw6/asnlib/publicdata/business_first.json"
#     second_json_path = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw6/asnlib/publicdata/business_second.json"
#     output_file_path = "../output/output_task1.csv"
# else:
#     print('wrong system type.')
#     sys.exit()

first_json_path = sys.argv[1]
second_json_path = sys.argv[2]
output_file_path = sys.argv[3]

class Bitmap():
    def __init__(self, list_size):
        """
        list_size - size of a list. the list consists of either 1 or 0
        """
        self.size = list_size // 31 + 1
        self.array = [0] * self.size
    
    def initialize(self, l):
        for x in l:
            self.setValue(x, 1)
    
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

def getSparkContext():
    conf = SparkConf() \
        .setAppName("task") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    return sc

def convertStrToInt(s):
    return int(binascii.hexlify(s.encode('utf8')), 16)

def generateHashs(m, num_hash, seed=RANDOM_SEED):
    """
    m - the length of the filter bit array
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

def checkCity(city, bitarray, hashs):
    """
    bitarray -  bloom filter bit array
    """
    if city == '':
        return 0
    int_city = convertStrToInt(city)
    positions = [h(int_city) for h in hashs]
    for p in positions:
        if bitarray.getValue(p) != 1:
            return 0
    return 1

def main():
    hashs = generateHashs(BIT_ARRAY_LENGTH, NUM_HASH)

    sc = getSparkContext()
    cities = sc.textFile(first_json_path) \
        .map(json.loads) \
        .map(lambda x: x['city']) \
        .filter(lambda x: x != '') \
        .distinct()
    ones_position = cities.map(lambda x: convertStrToInt(x)) \
        .flatMap(lambda x: [h(x) for h in hashs]) \
        .distinct() \
        .collect()

    bitarray = Bitmap(BIT_ARRAY_LENGTH)
    bitarray.initialize(ones_position)

    res = sc.textFile(second_json_path) \
        .map(json.loads) \
        .map(lambda x: x['city']) \
        .map(lambda x: checkCity(x, bitarray, hashs)) \
        .map(lambda x: str(x)) \
        .collect()

    output = ' '.join(res)
    with open(output_file_path, 'w', encoding='utf-8') as fp:
        fp.write(output)

if __name__ == "__main__":
    t1 = time.time()
    main()
    print('Duration: %fs.' % (time.time() - t1))
