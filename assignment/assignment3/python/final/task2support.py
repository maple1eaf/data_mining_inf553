class Bitmap():
    def __init__(self, list_size):
        """
        list_size - size of a list. the list consists of either 1 or 0
        """
        self.size = list_size // 31 + 1
        self.array = [0] * self.size
        self.count_one = 0
    
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
    
    def initialize(self, l, words_dict):
        self.count_one = len(l)
        for word in l:
            self.setValue(words_dict[word], 1)
    
    def setBitmap(self, size, count_one, array):
        self.size = size
        self.array = array
        self.count_one = count_one
    
    def countOneInt(self, integer):
        count = 0
        while(integer != 0):
            if integer & 1 == 1:
                count += 1
            integer = integer >> 1
        return count

    def countOne(self, array):
        sum = 0
        for i in range(len(array)):
            sum += self.countOneInt(array[i])
        return sum
    
    def countAndSetCountOne(self):
        self.count_one = self.countOne(self.array)
    
    def unionTwo(self, other):
        res_list = [0] * self.size
        for i in range(self.size):
            res_list[i] = self.array[i] | other.array[i]
        res_bitmap = Bitmap(self.size)
        res_bitmap.setBitmap(self.size, 0, res_list)
        return res_bitmap

    def intersectTwo(self, other):
        res_list = [0] * self.size
        for i in range(self.size):
            res_list[i] = self.array[i] & other.array[i]
        res_bitmap = Bitmap(self.size)
        res_bitmap.setBitmap(self.size, 0, res_list)
        # get number of ones
        res_bitmap.countAndSetCountOne()
        return res_bitmap