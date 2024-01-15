# @Time   : 2024.01.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from model.framework.lfu import LFU
from collections.abc import Iterable

class LRU(LFU):
    '''Least Recent Used Algorithm'''
    def __init__(self, cache_cfg, type) -> None:
        super().__init__(cache_cfg, type);
        self.recent_unique = True;

    def pop_item(self, n: int) -> None:
        if n == 0:
            return;
        for i in range(n):
            self.cache.remove(self.recent_used[i]);

