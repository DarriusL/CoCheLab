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
    
    def extend(self, cache: deque, __iterable: Iterable) -> None:
        for item in __iterable:
            if item in cache:
                __iterable.remove(item);
        return super().extend(cache, __iterable, is_unique = False)
        
    def update(self, seqs) -> None:
        ''''''
        self.extend(self.cache, seqs.tolist());
