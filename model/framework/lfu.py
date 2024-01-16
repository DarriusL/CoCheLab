# @Time   : 2024.01.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from model.framework.lru import LRU
from collections import deque, Counter, OrderedDict
import numpy as np

class LFU(LRU):
    '''Least Frequently Used Algorithm'''
    def __init__(self, cache_cfg) -> None:
        super().__init__(cache_cfg);
        self.recent_used = deque(maxlen = self.recent_used_n);
        self.recent_unique = False;
        self.cache_left = self.bs_storagy;
    
    def pop_item(self, n:int) -> None:
        if n == 0:
            return;
        c = OrderedDict.fromkeys(self.cache, 1);
        c_r = Counter(self.recent_used);
        for key in self.cache:
            if key in c_r.keys():
                c[key] += c_r[key];
        #Make sure that when the frequencies are the same, the data that comes in first will pop up.
        sort_indx = np.argsort(list(c.values()), kind = 'stable');
        for i in range(n):
            self.cache.remove(list(c.keys())[sort_indx[i]]);
            
    def update(self, seqs) -> None:
        ''''''
        seqs = seqs.tolist();
        n_unique = self.check_unique(seqs);
        if self.cache_left >= n_unique:
            self.cache_left -= n_unique;
        elif self.cache_left == 0:
            self.pop_item(n_unique);
        else:
            self.pop_item(n_unique - self.cache_left);
            self.cache_left = 0;
        self.extend(self.cache, seqs, self.cache_unique);
        self.extend(self.recent_used, seqs, self.recent_unique);