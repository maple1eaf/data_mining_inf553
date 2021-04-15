from pyspark import SparkConf, SparkContext
import time

input_file_path = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw2/dataset/small1.csv"

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



conf = SparkConf() \
    .setAppName("task1") \
    .setMaster("local[*]")
sc = SparkContext(conf=conf)

# data_with_header = sc.textFile(input_file_path)
# header = data_with_header.first()
# distinct_business_id_list = data_with_header.filter(lambda l: l != header) \
#     .map(lambda l: l.split(',')[1]) \
#     .distinct() \
#     .collect()
    
# business_pt = ProjectionTable(distinct_business_id_list)
# print(business_pt.iv)
# print(business_pt.getIndex("106"))
# print(business_pt.getIndex("10dfd"))
# print(business_pt.getValue(6))
# print(business_pt.getValue(20))
t1 = time.time()
data_with_header = sc.textFile(input_file_path)
header = data_with_header.first()
data_0 = data_with_header.filter(lambda l: l != header) \
    .map(lambda l: l.split(',')) \
    .cache()

distinct_data = data_0.map(lambda x: x[1]).distinct().collect()
data_pt = ProjectionTable(distinct_data)
print(data_pt.iv)

data_1 = data_0.groupByKey() \
    .map(lambda x: list(set(x[1]))) \
    .take(10)
print(data_1)

data = data_0.groupByKey() \
    .map(lambda x: list(set(x[1]))) \
    .map(lambda l: [data_pt.getIndex(x) for x in l]) \
    .cache()
print(data.take(10))
t2 = time.time()
print("use %fs." % (t2-t1))
