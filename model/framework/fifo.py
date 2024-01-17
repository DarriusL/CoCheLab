# @Time   : 2024.01.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from collections.abc import Iterable
from model.framework.base import Cache
from collections import deque

class FIFO(Cache):
    '''First in first out algorithm'''
    def __init__(self, cache_cfg) -> None:
        super().__init__(cache_cfg);
    
    def clear(self) -> None:
        self.cache.clear();
    
    def generate_subcache(self, n) -> set:
        if n >= len(self.cache):
            return set(self.cache);
        else:
            return set(list(self.cache)[-n:]);
    
    def  _extend(self, cache: deque, __iterable: Iterable) -> None:
        for item in __iterable:
            if item in cache:
                __iterable.remove(item);
        return super()._extend(cache, __iterable, is_unique = False)
        
    def update(self, seqs) -> None:
        '''

        Notes:
        ------
        elements of seqs do not exceed the bs_storagy
        '''
        self._extend(self.cache, seqs.tolist());
