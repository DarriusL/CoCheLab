# @Time   : 2024.01.15
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from model.framework.base import Cache
from model.framework.fifo import FIFO

class LRU(Cache):
    '''Least Recent Used Algorithm'''
    def __init__(self, cache_cfg) -> None:
        super().__init__(cache_cfg);
        self.cache_unique = True;
    
    def clear(self) -> None:
        return FIFO.clear(self);
    
    def generate_subcache(self, n) -> set:
        return FIFO.generate_subcache(self, n);
        
    def update(self, seqs) -> None:
        '''
        Notes:
        ------
        elements of seqs do not exceed the bs_storagy
        '''
        self._extend(self.cache, seqs.tolist(), self.cache_unique);

