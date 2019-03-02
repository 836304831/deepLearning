# -*- coding: utf-8 -*-
# -----------------------------------
# File: binary_classification_net.py
# Author: acedar
# Date: 2019.03.02
# -----------------------------------

import os
import tensorflow as tf
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
y = tf.placeholder(dtype=tf.int64, shape=[None], name="y")

w = tf.get_variable(name="w", shape=[x.get_shape()[-1], 1], initializer=tf.random_normal_initializer(0, 1))
b = tf.get_variable(name="b", shape=[1], initializer=tf.constant_initializer(0.0))

y_ = tf.matmul(x, w) + b
prob_y = tf.nn.sigmoid(y_)

y_reshaped = tf.reshape(y, shape=(-1, 1))
loss = tf.reduce_mean(tf.square(tf.cast(y_reshaped, tf.float32) - prob_y))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(prob_y > 0.5, tf.int64), y_reshaped), tf.float32))


# --------------- train ---------------------------------
train_data_manager = DataManager(train_file_list, need_shuffle=True)
train_data_manager.load_data()
train_data_manager.filter_binary_data()

test_data_manager = DataManager(test_file_list, need_shuffle=False)
test_data_manager.load_data()
test_x, test_y = test_data_manager.filter_binary_data()

variable_init = tf.global_variables_initializer()

batch_size = 20
iteration = 100000

"""iteration=100000, acc=81.75
   iteration=10000, acc=80.6
   iteration=100000, normal: [0, 1], acc=81.40
   iteration=10000, normal: [0, 1], acc=69.7
"""

with tf.Session() as sess:
    sess.run(variable_init)
    for i in range(iteration):
        batch_data, batch_labels = train_data_manager.next_batch(batch_size)

        acc_res, loss_res, _, prob_y_res = sess.run([accuracy, loss, train_op, prob_y], feed_dict={x: batch_data, y: batch_labels})

        if (i + 1) % 500 == 0:
            print("train step: %d, loss: %4.5f, acc: %4.4f" % ((i + 1), loss_res, acc_res))
        if (i + 1) % 2000 == 0:
            acc_res, loss_res, prob_y_res = sess.run([accuracy, loss, prob_y], feed_dict={x: test_x, y: test_y})
            print("*****test step: %d, loss: %4.5f, acc: %4.4f" % ((i + 1), loss_res, acc_res))
            # print("prob_y:\n{}".format(prob_y_res))
