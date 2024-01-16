# @Time   : 2024.01.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from model.framework.base import Cache
from collections.abc import Iterable

class LRU(Cache):
    '''Least Recent Used Algorithm'''
    def __init__(self, cache_cfg) -> None:
        super().__init__(cache_cfg);
        self.cache_unique = True;
        
    def update(self, seqs) -> None:
        self.extend(self.cache, [seqs.tolist()], self.cache_unique);

