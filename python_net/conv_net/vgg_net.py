# -*- coding: utf-8 -*-
# -----------------------------------
# File: binary_classification_net.py
# Author: acedar
# Date: 2019.03.02
# -----------------------------------

import os
import tensorflow as tf
import time
from python_net.common.data_manager import DataManager

__dir__ = os.path.abspath(os.path.dirname(__file__))
WORK_DIR = os.path.dirname(os.path.dirname(__dir__))
cifar10_path = os.path.join(WORK_DIR, "datasets/cifar-10-batches-py")

train_file_list = [os.path.join(cifar10_path, "data_batch_1"), os.path.join(cifar10_path, "data_batch_2"),
                   os.path.join(cifar10_path, "data_batch_3"), os.path.join(cifar10_path, "data_batch_4"),
                   os.path.join(cifar10_path, "data_batch_5")]

test_file_list = [os.path.join(cifar10_path, "test_batch")]


# ------------------- net build--------------------------------
x = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name="input")
x_image = tf.reshape(x, (-1, 3, 32, 32))
x_image = tf.transpose(x_image, [0, 2, 3, 1])
y = tf.placeholder(dtype=tf.int64, shape=[None], name="y")

conv1_1 = tf.layers.conv2d(x_image, 32, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv1_1")
conv1_2 = tf.layers.conv2d(conv1_1, 32, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv1_2")
max_pooling1 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name="max_pooling1")

conv2_1 = tf.layers.conv2d(max_pooling1, 32, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv2_1")
conv2_2 = tf.layers.conv2d(conv2_1, 32, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv2_2")
max_pooling2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name="max_pooling2")

conv3_1 = tf.layers.conv2d(max_pooling2, 32, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv3_1")
conv3_2 = tf.layers.conv2d(conv1_1, 32, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv3_2")
max_pooling3 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name="max_pooling3")


flatten = tf.layers.flatten(max_pooling3, name="flatten")
y_ = tf.layers.dense(flatten, 10, name="output")
# prob_y = tf.nn.softmax(y_)

# y_reshaped = tf.one_hot(y, 10, dtype=tf.float32)
# loss = tf.reduce_mean(tf.square(tf.cast(y_reshaped, tf.float32) - prob_y))
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


prob_y_ = tf.argmax(y_, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prob_y_, y), tf.float32))


# --------------- train ---------------------------------
train_data_manager = DataManager(train_file_list, need_shuffle=True)
train_data_manager.load_data()

test_data_manager = DataManager(test_file_list, need_shuffle=False)
test_x, test_y = test_data_manager.load_data()

variable_init = tf.global_variables_initializer()

batch_size = 20
iteration = 10000

"""iteration=100000, acc=61.17, time_consume: 252 (s)
   iteration=10000, acc=67.15, time_consume: 26 (s)
"""
with tf.Session() as sess:
    sess.run(variable_init)
    start_time = time.time()
    for i in range(iteration):
        batch_data, batch_labels = train_data_manager.next_batch(batch_size)

        acc_res, loss_res, _ = sess.run([accuracy, loss, train_op], feed_dict={x: batch_data, y: batch_labels})

        if (i + 1) % 500 == 0:
            print("train step: %d, loss: %4.5f, acc: %4.4f" % ((i + 1), loss_res, acc_res))
        if (i + 1) % 2000 == 0:
            acc_res, loss_res = sess.run([accuracy, loss], feed_dict={x: test_x, y: test_y})
            print("*****test step: %d, loss: %4.5f, acc: %4.4f" % ((i + 1), loss_res, acc_res))
    end_time = time.time()
    print("total consume time: {} (s)".format(end_time-start_time))
