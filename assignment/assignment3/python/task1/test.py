from pyspark import SparkConf, SparkContext


# define spark env
conf = SparkConf() \
    .setAppName("task1") \
    .setMaster("local[*]")

sc = SparkContext(conf=conf)

list_a = [('a1',[1,2,3]), ('a2',[5,6])]
list_b = [('b1',[11,12,13]), ('b2',[25,26,28])]

rdd_a = sc.parallelize(list_a)
rdd_b = sc.parallelize(list_b)

cart = rdd_a.cartesian(rdd_b).collect()

print(cart)

"""
result:
[
    (('a1', [1, 2, 3]), ('b1', [11, 12, 13])), 
    (('a1', [1, 2, 3]), ('b2', [25, 26, 28])), 
    (('a2', [5, 6]), ('b1', [11, 12, 13])), 
    (('a2', [5, 6]), ('b2', [25, 26, 28]))
]
"""

