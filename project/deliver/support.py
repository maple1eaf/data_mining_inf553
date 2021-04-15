import random
import json

class Als():
    def __init__(self):
        self.user_feature = None
        self.product_feature = None
    
    def setModel(self, user_feature, product_feature):
        """
        user_feature - {uid: float, ...}
        product_feature - {bid: float, ...}
        """
        self.user_feature = user_feature
        self.product_feature = product_feature
    
    def predict(self, uid, bid):
        u_value = self.user_feature.get(uid)
        b_value = self.product_feature.get(bid)
        if u_value == None or b_value == None:
            return None
        res = u_value * b_value
        # if res > 5.0:
        #     res = 5.0
        # elif res < 0.0:
        #     res = 0.0
        return res

def getAvg(avg_file):
    with open(avg_file, 'r', encoding='utf-8') as fp:
        avg_d = json.load(fp)
    return avg_d

def writeDownRenameTable(table, file_path):
    with open(file_path, 'w', encoding='utf-8') as fp:
        json.dump(table, fp)

def readRenameTable(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        table = json.load(fp)
    return table

def saveBusinessSim(js_list, js_file):
    with open(js_file, 'w', encoding='utf-8') as fp:
        json.dump(js_list, fp)

def loadBusinessSim(js_file):
    with open(js_file, 'r', encoding='utf-8') as fp:
        jaccard_sim = json.load(js_file)
    return jaccard_sim


def meanList(l):
    return sum(l) / len(l)

