"""
Python: $ spark-submit task1.py <input_file> <output_file> <stopwords> <y> <m> <n>
spark-submit task1.py "file:////Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json" "output_task1.json" "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/stopwords" 2018 5 5
"""

import json
import re
import sys

from pyspark import SparkContext

# caution: y, m, n are str
input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords_file_path = sys.argv[3]
y = sys.argv[4]
m = sys.argv[5]
n = sys.argv[6]

# define spark env
sc = SparkContext(master="local[*]", appName="task1")

# parameter
data_file_path = input_file
# read on local disk
# data_file_path = "file:////Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw1/hw1_data/review.json"
# read on hdfs
# data_file_path = "hw1/hw1_data/review.json"

# load data, convert to object in python
review_data = sc.textFile(data_file_path).map(json.loads).cache()

results = {}

# A
total_number_of_reviews = review_data.count()
results['A'] = total_number_of_reviews

# B
number_of_reviews_in_a_given_year = review_data.filter(lambda r: r['date'][:4]==y).count()
results['B'] = number_of_reviews_in_a_given_year

# C
number_of_distinct_users = review_data.map(lambda r: r['user_id']).distinct().count()
results['C'] = number_of_distinct_users

# D
m = int(m)
top_m_users = review_data.map(lambda r: (r['user_id'], 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .sortBy(keyfunc=lambda x: x[0], ascending=True) \
    .sortBy(keyfunc=lambda x: x[1], ascending=False) \
    .take(m)
results['D'] = top_m_users

# E
n = int(n)
PUNCTUATIONS_PATERN = r"[\(\[,.!?:;\]\)\s]"
stopwords_path = stopwords_file_path
stopwords = sc.textFile(stopwords_path).collect()

def exculdePuncsAndWords(r):
    # get text value from a record
    text = r['text']
    # lower case
    text_lower = text.lower()
    # divide by punctuations and space
    words_list = re.split(PUNCTUATIONS_PATERN, text_lower)
    # delete stopwords and null charactor
    words_list_clean_as_key_with_1 = [[word,1] for word in words_list if word != '' and word not in stopwords]
    return words_list_clean_as_key_with_1

top_n_frequent_words = review_data.flatMap(exculdePuncsAndWords) \
    .reduceByKey(lambda x, y: x + y) \
    .sortBy(keyfunc=lambda x: x[0], ascending=True) \
    .sortBy(keyfunc=lambda x: x[1], ascending=False) \
    .map(lambda x: x[0]) \
    .take(n)
results['E'] = top_n_frequent_words

# write on local disk instead of hdfs
with open(output_file, 'w', encoding='utf-8') as fp:
    json.dump(results, fp)
