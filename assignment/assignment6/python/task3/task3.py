"""
spark-submit task3.py <port #> <output_file_path>

spark-submit task3.py 9999 "../output/output_task3.csv"
"""

import sys
import os
import time
import json
import random

from pyspark import SparkConf, SparkContext, StorageLevel
import tweepy

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

CONSUMER_TOKEN = "hY3hLVZSsujafToV1NoWiIslF"
CONSUMER_SECRET = "w8t97BvBg6frWFKcRFh3xKBw5YP0l86mF8sUll2HPelkWFxc8P"
ACCESS_TOKEN = "1043393088331247617-coRiLixUNXdHVaIRavzKbpBVaSPdqL"
ACCESS_TOKEN_SECRET = "Y7wcLcVb5OFH8x5BKHkfBTWopAGM5nkFr2Zv5IlXUY4cJ"

SAMPLE_SIZE = 100
TOP_FREQUENCIES = 3

port_num = int(sys.argv[1])
output_file_path = sys.argv[2]

class ReservoirSampling():
    def __init__(self, size):
        """
        self.size: int - sample size
        self.count: int - count from the beginning
        """
        self.size = size
        self.sample = []
        self.count = 0
        
    def decide(self, element):
        if self.size > len(self.sample):
            self.sample.append(element)
            self.count += 1
        else:
            if random.random() < self.size / (self.count + 1):
                del_idx = random.randint(0, self.size - 1)
                self.sample.pop(del_idx)
                self.sample.append(element)
            self.count += 1

    def writeToFile(self, vk_candi):
        with open(output_file_path, 'a', encoding='utf-8') as fp:
            fp.write('The number of tweets with tags from the beginning: %d\n' % (self.count))
            for item in vk_candi:
                for tag in item[1]:
                    fp.write('%s  : %d\n' % (tag, item[0]))
            fp.write('\n')
    
    def findPopularTags(self, sc):
        vk_candi = sc.parallelize(self.sample) \
            .flatMap(lambda x: [(item, 1) for item in x]) \
            .reduceByKey(lambda x, y: x + y) \
            .map(lambda x: (x[1], [x[0]])) \
            .reduceByKey(lambda x, y: x + y) \
            .map(lambda x: (x[0], sorted(x[1]))) \
            .sortBy(keyfunc=lambda x: x[0], ascending=False) \
            .take(TOP_FREQUENCIES)
        self.writeToFile(vk_candi)

class TwitterListener(tweepy.StreamListener):
    def __init__(self, sample, sc):
        super(TwitterListener, self).__init__()
        self.sample = sample
        self.sc = sc

    def on_status(self, status):
        if status.entities['hashtags'] == []:
            return True
        tags = status.entities['hashtags']
        for tag in tags:
            if not isEnglish(tag['text']):
                return True
        texts = []
        for tag in tags:
            texts.append(tag['text'])
        self.sample.decide(texts)
        self.sample.findPopularTags(self.sc)

    
    def on_error(self, status_code):
        # returning non-False reconnects the stream, with backoff.
        return True

def getAndSendTwitterTags(sc):
    auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    sample = ReservoirSampling(SAMPLE_SIZE)
    stream_listener = TwitterListener(sample, sc)
    twitter_stream = tweepy.Stream(auth = auth, listener=stream_listener)
    twitter_stream.api.wait_on_rate_limit = True
    twitter_stream.filter(track=['covid'])

def isEnglish(s):
    try:
        return all(ord(c) < 128 for c in s)
    except TypeError:
        return False

def main():
    conf = SparkConf() \
        .setAppName("task") \
        .setMaster("local[*]") \
        .set("spark.driver.memory","4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    getAndSendTwitterTags(sc)

if __name__ == "__main__":
    main()



