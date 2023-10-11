from typing import Dict, List, Tuple
import numpy as np
import random
from collections    import deque
from operator       import itemgetter


class Buffer(object):
    def __init__(self, buffer_size: int) -> None:
        super(Buffer, self).__init__()
        self.size = buffer_size
        self.data = deque(maxlen=self.size)
    
    def store(self, trans: Tuple) -> None:
        self.data.append(trans)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.data, batch_size)
        data_batch = []

        for item in zip(*batch):
            item = np.stack(item, 0)
            data_batch.append(item)

        data_batch = tuple(data_batch)
        return data_batch

    def sample_duplicated_batch(self, batch_size: int) -> Tuple:
        idxes = np.random.randint(0, len(self.data), batch_size)
        batch = list(itemgetter(*idxes)(self.data))
        data_batch = []
        for item in zip(*batch):
            item = np.stack(item, 0)
            data_batch.append(item)
            
        data_batch = tuple(data_batch)
        return data_batch

    def sample_all(self) -> Tuple:
        data_batch = []
        for item in zip(*self.data):
            item = np.stack(item, 0)
            data_batch.append(item)
        return tuple(data_batch)

    def clear(self) -> None:
        self.data.clear()

    def __len__(self):
        return len(self.data)