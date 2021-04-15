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

def meanList(l):
    return sum(l) / len(l)

def averageRating(x):
    uid = x[0][0]
    bid = x[0][1]
    stars = x[1] # list
    average_stars = meanList(stars)
    return (uid, [(bid, average_stars)])

def convertToDict(l):
    # l - [(bid, star), ...]
    bs = {}
    for bid, star in l:
        bs[bid] = star
    return bs












def isPrime(num):
    if num == 1:
        return False
    for divider in range(2, num):
        if num % divider == 0:
            return False
    return True

def smallestPrime(num):
    while(1):
        num = num + 1
        if isPrime(num):
            return num

def generateHashs(n, m):
    """
    generate a list of hash functions
    :n - number of hash functions we want to generate
    :m - number of attributes
    """
    random.seed(12)
    ab = []
    for i in range(2*n):
        n1 = random.randint(1,m)
        if n1 not in ab:
            ab.append(n1)

    a_list = ab[:n]
    b_list = ab[n:]
    m_new = smallestPrime(m)
    
    # print('ab list length:', len(ab))
    # print('the prime number m_new:', m_new)

    def generateHash(i):
        a = a_list[i]
        b = b_list[i]
        def hashfunc(x):
            return (a * hash(x) + b) % m_new
        return hashfunc
    return [generateHash(i) for i in range(n)]

def generateHashForLSH(r):
    random.seed(8)
    l_ = []
    for i in range(r):
        a = random.randint(1, 10000)
        if a not in l_:
            l_.append(a)

    def h(l):
        # l - list of integers
        l_len = len(l)
        sum_ = 0
        for i in range(l_len):
            sum_ = sum_ + l_[i] * l[i]
        return sum_ % 50000
    return h

