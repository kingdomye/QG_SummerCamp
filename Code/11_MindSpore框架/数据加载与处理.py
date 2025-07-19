import numpy as np
from mindspore.dataset import GeneratorDataset


# Random-accessible object as input source
class RandomAccessDataset:
    def __init__(self):
        self._data = np.ones((5, 2))
        self._label = np.zeros((5, 1))

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


# Iterator as input source
class IterableDataset:
    def __init__(self, start, end):
        """init the class object to hold the data"""
        self.start = start
        self.end = end

    def __next__(self):
        """iter one data and return"""
        return next(self.data)

    def __iter__(self):
        """reset the iter"""
        self.data = iter(range(self.start, self.end))
        return self


loader = RandomAccessDataset()
dataset = GeneratorDataset(source=loader, column_names=["data", "label"])

for data in dataset:
    print(data)

loader = IterableDataset(1, 5)
dataset = GeneratorDataset(source=loader, column_names=["data"])

for d in dataset:
    print(d)
