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
from python_net.common import utils

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
conv3_2 = tf.layers.conv2d(conv3_1, 32, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv3_2")
max_pooling3 = tf.layers.max_pooling2d(conv3_2, [2, 2], [2, 2], name="max_pooling3")


flatten = tf.layers.flatten(max_pooling3, name="flatten")
y_ = tf.layers.dense(flatten, 10, name="output")
# prob_y = tf.nn.softmax(y_)

# y_reshaped = tf.one_hot(y, 10, dtype=tf.float32)
# loss = tf.reduce_mean(tf.square(tf.cast(y_reshaped, tf.float32) - prob_y))
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


prob_y_ = tf.argmax(y_, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prob_y_, y), tf.float32))

# tensorboard summary

loss_summary = tf.summary.scalar("loss", loss)
acc_summary = tf.summary.scalar("accuracy", accuracy)
# conv1_1_summary = tf.summary.scalar("summary/conv1_1", conv1_1)

train_summary = tf.summary.merge_all()
test_summary = tf.summary.merge([loss_summary, acc_summary])

source_image = (x_image + 1) + 127.5
image_summary = tf.summary.image("input_image", source_image)

LOG_DIR = os.path.join(WORK_DIR, "log")

train_log_path = utils.path_generate(LOG_DIR, "train")
test_log_path = utils.path_generate(LOG_DIR, "test")

# --------------- train ---------------------------------
train_data_manager = DataManager(train_file_list, need_shuffle=True)
train_data_manager.load_data()

test_data_manager = DataManager(test_file_list, need_shuffle=False)
test_x, test_y = test_data_manager.load_data()

variable_init = tf.global_variables_initializer()

batch_size = 16
iteration = 10000

"""
    iteration=100000, acc=73.01, time_consume: 375 (s)
    iteration=10000, acc=71.48, time_consume: 39 (s)
"""
with tf.Session() as sess:
    sess.run(variable_init)
    start_time = time.time()
    train_summary_writer = tf.summary.FileWriter(train_log_path, sess.graph)
    test_summary_writer = tf.summary.FileWriter(test_log_path)

    for i in range(iteration):
        batch_data, batch_labels = train_data_manager.next_batch(batch_size)
        if (i + 1) % 100 == 0:
            acc_res, loss_res, _, train_summary_res = sess.run([accuracy, loss, train_op, train_summary],
                                                               feed_dict={x: batch_data, y: batch_labels})
            train_summary_writer.add_summary(train_summary_res, i + 1)
            test_acc_res, test_loss_res, test_summary_res = sess.run([accuracy, loss, test_summary], feed_dict={x: test_x, y: test_y})
            test_summary_writer.add_summary(test_summary_res, i + 1)
        else:
            acc_res, loss_res, _ = sess.run([accuracy, loss, train_op], feed_dict={x: batch_data, y: batch_labels})
        if (i + 1) % 500 == 0:
            print("train step: %d, loss: %4.5f, acc: %4.4f" % ((i + 1), loss_res, acc_res))
        if (i + 1) % 2000 == 0:
            acc_res, loss_res = sess.run([accuracy, loss], feed_dict={x: test_x, y: test_y})
            print("*****test step: %d, loss: %4.5f, acc: %4.4f" % ((i + 1), loss_res, acc_res))
    train_summary_writer.close()
    test_summary_writer.close()
    end_time = time.time()
    print("total consume time: {} (s)".format(end_time-start_time))
