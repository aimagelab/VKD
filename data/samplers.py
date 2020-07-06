from collections import defaultdict
from itertools import chain

import numpy as np
import random

from torch.utils.data.sampler import Sampler


def compute_pids_and_pids_dict(data_source):

    index_dic = defaultdict(list)
    for index, (_, pid, _) in enumerate(data_source):
        index_dic[pid].append(index)
    pids = list(index_dic.keys())
    return pids, index_dic


class ReIDBatchSampler(Sampler):

    def __init__(self, data_source, p: int, k: int):

        self._p = p
        self._k = k

        pids, index_dic = compute_pids_and_pids_dict(data_source)

        self._unique_labels = np.array(pids)
        self._label_to_items = index_dic.copy()

        self._num_iterations = len(self._unique_labels) // self._p

    def __iter__(self):

        def sample(set, n):
            if len(set) < n:
                return np.random.choice(set, n, replace=True)
            return np.random.choice(set, n, replace=False)

        np.random.shuffle(self._unique_labels)

        for k, v in self._label_to_items.items():
            random.shuffle(self._label_to_items[k])

        curr_p = 0

        for idx in range(self._num_iterations):
            p_labels = self._unique_labels[curr_p: curr_p + self._p]
            curr_p += self._p
            batch = [sample(self._label_to_items[l], self._k) for l in p_labels]
            batch = list(chain(*batch))
            yield batch

    def __len__(self):
        return self._num_iterations
