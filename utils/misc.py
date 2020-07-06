import argparse
from collections import defaultdict
from typing import List

import numpy as np


def init_worker(worker_id):
    np.random.seed(1234 + worker_id)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class SingleAvgMeter:

    def __init__(self, label: str):
        self.values = []
        self.label = label

    def add(self, value: float):
        self.values.append(value)

    def __call__(self):
        return (self.label, np.array(self.values).mean())

    def reset(self):
        self.values = ()


class AvgMeter:

    def __init__(self, labels: List[str]):

        self.avg_meters = []

        for i in range(len(labels)):
            self.avg_meters.append(SingleAvgMeter(labels[i]))

    def add(self, values: List[float]):
        for i, v in enumerate(values):
            self.avg_meters[i].add(v)

    def __call__(self):
        return [avg_meter() for avg_meter in self.avg_meters]

    def reset(self):
        for avg_meter in self.avg_meters:
            avg_meter.reset()
