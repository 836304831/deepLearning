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

w = tf.get_variable(name="w", shape=[x.get_shape()[-1], 10], initializer=tf.random_normal_initializer(0, 1))
b = tf.get_variable(name="b", shape=[10], initializer=tf.constant_initializer(0.0))

y_ = tf.matmul(x, w) + b
prob_y = tf.nn.softmax(y_)

y_reshaped = tf.one_hot(y, 10, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(tf.cast(y_reshaped, tf.float32) - prob_y))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


prob_y_ = tf.argmax(prob_y, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prob_y_, y), tf.float32))


# --------------- train ---------------------------------
train_data_manager = DataManager(train_file_list, need_shuffle=True)
train_data_manager.load_data()

test_data_manager = DataManager(test_file_list, need_shuffle=False)
test_x, test_y = test_data_manager.load_data()

variable_init = tf.global_variables_initializer()

batch_size = 20
iteration = 100000

"""iteration=100000, acc=39.42
   iteration=10000, acc=33.26
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
