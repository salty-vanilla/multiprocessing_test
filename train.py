from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import sys
import tensorflow as tf

from mnist import load_mnist
import sys

FLAGS = None

batch_size = 100
nb_epochs = 10
verbose = 1
valid_num = 10000

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def main():
    # # Arg parse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='./input_data/',
    #                     help='Type mnist data path')
    # parser.add_argument('--verbose', type=int, default=1)
    #
    # args = parser.parse_args()
    #
    # data_dir = args.data_dir
    # verbose = args.verbose

    # Import data
    train_csv = "./data/train.csv"
    valid_csv = "./data/valid.csv"
    test_csv = "./data/test.csv"

    mnist = load_mnist(train_csv, valid_csv, test_csv)

    # Create the model
    with tf.variable_scope('input'):
        inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])

    with tf.variable_scope('conv1'):
        W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
        b_conv1 = tf.Variable(tf.random_normal([32]))
        h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv2'):
        W_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
        b_conv2 = tf.Variable(tf.random_normal([64]))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope('fc'):
        flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        with tf.variable_scope('fc1'):
            W_fc1 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))
            b_fc1 = tf.Variable(tf.random_normal([1024]))
            h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)

        with tf.variable_scope('fc2'):
            W_fc2 = tf.Variable(tf.random_normal([1024, 10]))
            b_fc2 = tf.Variable(tf.random_normal([10]))
            outputs = tf.matmul(h_fc1, W_fc2) + b_fc2


    # Label placeholder
    labels = tf.placeholder(tf.float32, [None, 10])

    # Correct
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Loss and Optimizer
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train
    for epoch in range(nb_epochs):
        sys.stdout.write("Epoch %d/%d\n" % (epoch + 1, nb_epochs))
        for iter, (batch_xs, batch_ys) in enumerate(mnist.train.next_batch(batch_size)):
            feed_dict = {inputs: batch_xs, labels: batch_ys}
            _, train_loss, train_accuracy = sess.run([train_step, cross_entropy, accuracy],
                                                     feed_dict=feed_dict)

            if verbose == 1:
                length = 30
                percentage = float(iter * batch_size / mnist.train.num_data)
                bar = "[" + "=" * int(length * percentage) + "-" * (length - int(length * percentage)) + "]"
                display = "\r{} / {} {} " \
                          "loss: {:.4f} - acc: {:.4f}" \
                    .format(iter * batch_size, mnist.train.num_data, bar, train_loss, train_accuracy)
                sys.stdout.write(display)
                sys.stdout.flush()

        # validation
        valid_loss = 0
        valid_accuracy = 0
        for iter, (batch_xs, batch_ys) in enumerate(mnist.valid.next_batch(batch_size)):
            feed_dict = {inputs: batch_xs, labels: batch_ys}
            v_l, v_a = sess.run([cross_entropy, accuracy],
                                feed_dict=feed_dict)
            valid_loss += v_l * len(batch_xs)
            valid_accuracy += v_a * len(batch_ys)
        valid_loss /= valid_num
        valid_accuracy /= valid_num

        if verbose == 1:
            display = " - val_loss : {:.4f} - val_acc : {:.4f}\n" \
                .format(valid_loss, valid_accuracy)
            sys.stdout.write(display)

    sys.stdout.write("\nComplete training !!\n")

    # Test trained model
    feed_dict = {inputs: mnist.train.images, labels: mnist.train.labels}
    test_loss, test_accuracy = sess.run([cross_entropy, accuracy],
                                        feed_dict=feed_dict)

    display = "test_loss : {:.4f} - test_acc : {:.4f}\n" \
        .format(test_loss, test_accuracy)
    sys.stdout.write(display)


if __name__ == '__main__':
    main()

