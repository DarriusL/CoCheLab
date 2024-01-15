# @Time   : 2024.01.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from model.framework.fifo import FIFO
from collections import deque, Counter
import numpy as np

class LFU(FIFO):
    '''Least Frequently Used Algorithm'''
    def __init__(self, cache_cfg, type) -> None:
        super().__init__(cache_cfg, type);
        self.cache_unique = True;
        self.recent_used = deque(maxlen = self.recent_used_n);
        self.recent_unique = False;
        self.cache_left = self.bs_storagy;
    
    def pop_item(self, n:int) -> None:
        if n == 0:
            return;
        c = Counter(self.recent_used);
        sort_indx = np.argsort(list(c.values()));
        for i in range(n):
            self.cache.remove(list(c.keys())[sort_indx[i]]);
            
    def update(self, seqs) -> None:
        ''''''
        seqs = seqs.tolist();
        self.extend(self.recent_used, seqs, self.recent_unique);
        n_unique = self.check_unique(seqs);
        if self.cache_left >= n_unique:
            self.cache_left -= n_unique;
        elif self.cache_left == 0:
            self.pop_item(n_unique);
        else:
            self.pop_item(n_unique - self.cache_left);
            self.cache_left = 0;
        self.extend(self.cache, seqs, self.cache_unique);