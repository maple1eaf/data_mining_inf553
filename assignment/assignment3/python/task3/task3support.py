import random

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

class Rename():
    """
    every value should be distinct
    """
    def __init__(self, values):
        """
        values - a list of values
        """
        self.values = list(values)
        self.values_length = len(values)
    
    def getNewValue(self, original_value):
        try:
            new_value = self.values.index(original_value)
        except ValueError:
            return None
        return new_value
    
    def getOriginalValue(self, new_value):
        try:
            original_value = self.values[new_value]
        except IndexError:
            return None
        return original_value