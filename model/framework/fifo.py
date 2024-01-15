# @Time   : 2024.01.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from model.framework.base import Cache
from collections import deque

class FIFO(Cache):
    '''First in first out algorithm'''
    def __init__(self, cache_cfg, type) -> None:
        super().__init__(cache_cfg, type);
        

    def update(self, seqs) -> None:
        ''''''
        self.extend(self.cache, seqs.tolist(), is_unique = True);
