import pickle
import collections


class LRUCache:

    def __init__(self, capacity=1000):
        self.dict = collections.OrderedDict()
        self.remain = capacity

    def load(self, path):
        with open(path) as f:
            state = (self.dict, self.remain)
            pickle.dump(state, f)

    def save(self, path):
        with open(path) as f:
            self.dict, self.remain = pickle.load(f)

    def get(self, key):
        if key not in self.dict:
            return -1
        v = self.dict.pop(key)
        self.dict[key] = v   # set key as the newest one
        return v

    def set(self, key, value):
        if key in self.dict:
            self.dict.pop(key)
        else:
            if self.remain > 0:
                self.remain -= 1
            else:  # self.dic is full
                self.dict.popitem(last=False)
        self.dict[key] = value

    def is_empty(self):
        return len(self.dict) == 0
