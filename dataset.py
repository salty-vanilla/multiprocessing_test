import numpy as np
from collections import Counter
import random
import csv
from PIL import Image


class Dataset(object):
    def __init__(self, src_csv, one_hot=True, normalization=False, color=True):
        self.src_csv = src_csv
        self.one_hot = one_hot
        self.normalization = normalization
        self.color = color

        f = open(src_csv, 'r')
        reader = csv.reader(f)
        data = np.array([line for line in reader])

        self.img_paths = data[:, 0]
        self.labels = np.array(list(map(int, data[:, 1])))

        self.num_data = len(self.img_paths)

        if self.one_hot:
            self.labels = self.convert_to_one_hot(self.labels)

        assert len(self.img_paths) == len(self.labels)

    @staticmethod
    def convert_to_one_hot(labels):
        counter = Counter(labels)
        label_num = len(list(counter.keys()))

        one_hot = np.zeros((labels.shape[0], label_num))

        for label, oh in zip(labels, one_hot):
            oh.put(label, 1)

        return one_hot

    def next_batch(self, batch_size, shuffle=True):
        indexes = [i for i in range(0, self.num_data)]
        max_iter = int(self.num_data / batch_size)

        if shuffle:
            random.shuffle(indexes)

        for iter in range(max_iter):
            batch_indexes = indexes[batch_size * iter: batch_size * (iter + 1)]

            if self.color:
                images = np.array([np.array(Image.open(path).convert('RGB'))
                                   for path in self.img_paths[batch_indexes]])

            else:
                images = np.array([np.array(Image.open(path).convert('L'))
                                   for path in self.img_paths[batch_indexes]])
                images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

            images = images.astype('float32')

            if self.normalization:
                images /= 255

            yield images, self.labels[batch_indexes]

        if not self.num_data == batch_size * (iter + 1):
            batch_indexes = indexes[batch_size * (iter + 1):]
            if self.color:
                images = np.array([np.array(Image.open(path).convert('RGB'))
                                   for path in self.img_paths[batch_indexes]])

            else:
                images = np.array([np.array(Image.open(path).convert('L'))
                                   for path in self.img_paths[batch_indexes]])
                images = images.reshape((images.shape[0], images.shape[1], images.shape[2], 1))

            images = images.astype('float32')

            if self.normalization:
                images /= 255
            yield images, self.labels[batch_indexes]


