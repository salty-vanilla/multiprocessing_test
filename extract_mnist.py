# coding:utf-8
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import os
import csv
from PIL import Image
import numpy as np


origin = "https://s3.amazonaws.com/img-datasets/mnist.pkl.gz"
valid_num = 10000

def get_file(url, dst_path):
    import urllib
    # noinspection PyBroadException
    try:
        urllib.urlretrieve(url, dst_path)
    except:
        from urllib import request
        request.urlretrieve(url, dst_path)


def extract_mnist(src_dir="./", dst_dir="data"):
    src_path = os.path.join(src_dir, "mnist.pkl.gz")

    if not os.path.exists(src_path):
        os.makedirs(src_dir, exist_ok=True)
        get_file(origin, src_path)

    f = gzip.open(src_path, 'rb')
    data = pickle.load(f, encoding='bytes')

    (x_train, y_train), (x_test, y_test) = data

    dst_dir_train = os.path.join(dst_dir, "train/")
    dst_dir_valid = os.path.join(dst_dir, "valid/")
    dst_dir_test = os.path.join(dst_dir, "test/")

    os.makedirs(dst_dir_train, exist_ok=True)
    f = open(os.path.join(dst_dir, "train.csv"), 'w', newline='')
    writer = csv.writer(f)
    for index, (x, y) in enumerate(zip(x_train[valid_num:], y_train[valid_num:])):
        img = Image.fromarray(np.uint8(x))
        dst_path = os.path.join(dst_dir_train, "{0:05d}.png".format(index))
        img.save(dst_path)
        writer.writerow([dst_path, y])
    f.close()

    os.makedirs(dst_dir_valid, exist_ok=True)
    f = open(os.path.join(dst_dir, "valid.csv"), 'w', newline='')
    writer = csv.writer(f)
    for index, (x, y) in enumerate(zip(x_train[:valid_num], y_train[:valid_num])):
        img = Image.fromarray(np.uint8(x))
        dst_path = os.path.join(dst_dir_valid, "{0:05d}.png".format(index))
        img.save(dst_path)
        writer.writerow([dst_path, y])
    f.close()

    os.makedirs(dst_dir_test, exist_ok=True)
    f = open(os.path.join(dst_dir, "test.csv"), 'w', newline='')
    writer = csv.writer(f)
    for index, (x, y) in enumerate(zip(x_test, y_test)):
        img = Image.fromarray(np.uint8(x))
        dst_path = os.path.join(dst_dir_test, "{0:05d}.png".format(index))
        img.save(dst_path)
        writer.writerow([dst_path, y])
    f.close()


extract_mnist()
