from collections import namedtuple
from dataset import Dataset
import os

def load_mnist(train_csv, valid_csv, test_csv,
               flatten=True, one_hot=True, normalization=True, color=False):
    datasets = namedtuple('Datasets', ['train', 'valid', 'test'])

    datasets.train = Dataset(src_csv=train_csv, one_hot=one_hot, normalization=normalization, color=color)
    datasets.valid = Dataset(src_csv=valid_csv, one_hot=one_hot, normalization=normalization, color=color)
    datasets.test = Dataset(src_csv=test_csv, one_hot=one_hot, normalization=normalization, color=color)

    return datasets
