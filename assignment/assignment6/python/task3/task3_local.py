"""
spark-submit task3.py <port #> <output_file_path>
Access token: 1043393088331247617-coRiLixUNXdHVaIRavzKbpBVaSPdqL
Access token secret: Y7wcLcVb5OFH8x5BKHkfBTWopAGM5nkFr2Zv5IlXUY4cJ
"""

import sys
import os
import time
import json
import random

import tweepy

CONSUMER_TOKEN = "hY3hLVZSsujafToV1NoWiIslF"
CONSUMER_SECRET = "w8t97BvBg6frWFKcRFh3xKBw5YP0l86mF8sUll2HPelkWFxc8P"
ACCESS_TOKEN = "1043393088331247617-coRiLixUNXdHVaIRavzKbpBVaSPdqL"
ACCESS_TOKEN_SECRET = "Y7wcLcVb5OFH8x5BKHkfBTWopAGM5nkFr2Zv5IlXUY4cJ"

SAMPLE_SIZE = 100
TOP_FREQUENCIES = 3

port_num = 9999
output_file_path = "../output/output_task3.csv"

class ReservoirSampling():
    def __init__(self, size):
        """
        self.size: int - sample size
        self.count: int - count from the beginning
        """
        self.size = size
        self.sample = []
        self.count = 0
        self._popular_tags = {}
        self.new_item = []
        self.discard_item = []
        
    def decide(self, element):
        if self.size > len(self.sample):
            self.new_item = element
            self.sample.append(element)
            self.count += 1
        else:
            if random.random() < self.size / (self.count + 1):
                self.new_item = element
                del_idx = random.randint(0, self.size - 1)
                self.discard_item = self.sample.pop(del_idx)
                self.sample.append(element)
            self.count += 1

    def writeToFile(self, vk_candi):
        with open(output_file_path, 'a', encoding='utf-8') as fp:
            fp.write('The number of tweets with tags from the beginning: %d\n' % (self.count))
            for item in vk_candi:
                for tag in item[1]:
                    fp.write('%s  : %d\n' % (tag, item[0]))
            fp.write('\n')

    def findPopularTags(self):
        if self._popular_tags == {}:
            for tags in self.sample:
                for tag in tags:
                    if self._popular_tags.get(tag) == None:
                        self._popular_tags[tag] = 1
                    else:
                        self._popular_tags[tag] += 1
        else:
            for tag in self.new_item:
                if self._popular_tags.get(tag) == None:
                    self._popular_tags[tag] = 1
                else:
                    self._popular_tags[tag] += 1
            for tag in self.discard_item:
                self._popular_tags[tag] -= 1
                if self._popular_tags[tag] == 0:
                    self._popular_tags.pop(tag, None)
        
        vk = {}
        for k in self._popular_tags:
            v = self._popular_tags[k]
            if vk.get(v) == None:
                vk[v] = [k]
            else:
                vk[v].append(k)
        
        vk_list = list(vk.items())
        vk_candi = sorted(vk_list, key=lambda x: x[0], reverse=True)[:TOP_FREQUENCIES]
        for item in vk_candi:
            item[1].sort()
        self.writeToFile(vk_candi)


class TwitterListener(tweepy.StreamListener):
    def __init__(self, sample):
        super(TwitterListener, self).__init__()
        self.sample = sample

    def on_status(self, status):
        if status.entities['hashtags'] == []:
            return
        tags = status.entities['hashtags']
        texts = []
        for tag in tags:
            texts.append(tag['text'])
        self.sample.decide(texts)
        self.sample.findPopularTags()
    
    def on_error(self, status_code):
        # returning non-False reconnects the stream, with backoff.
        return True

def getAndSendTwitterTags():
    auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    sample = ReservoirSampling(SAMPLE_SIZE)
    stream_listener = TwitterListener(sample)
    twiter_stream = tweepy.Stream(auth = auth, listener=stream_listener)
    twiter_stream.filter(track=['covid'])





def main():
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    getAndSendTwitterTags()


if __name__ == "__main__":
    main()



