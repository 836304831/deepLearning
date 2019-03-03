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


# mobile_net block
def mobile_net_block(input, output_channel, name):
    with tf.variable_scope(name):
        input_channel = input.get_shape().as_list()[3]
        splited_channel_list = tf.split(input, input_channel, axis=3)
        output_channel_list = []
        for i in range(input_channel):
            one_channel = tf.layers.conv2d(splited_channel_list[i], 1, (3, 3), (1, 1), padding="same",
                                           activation=tf.nn.relu, name="conv_%d" % i)
            output_channel_list.append(one_channel)
        conv_by_channel = tf.concat(output_channel_list, axis=3)
        output_conv = tf.layers.conv2d(conv_by_channel, output_channel, (1, 1), (1, 1), padding="same",
                                       activation=tf.nn.relu, name="conv_by_channel")
        return output_conv


# ------------------- net build--------------------------------
x = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name="input")
x_image = tf.reshape(x, (-1, 3, 32, 32))
x_image = tf.transpose(x_image, [0, 2, 3, 1])
y = tf.placeholder(dtype=tf.int64, shape=[None], name="y")


conv_pre = tf.layers.conv2d(x_image, 16, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv_pre")
max_pool_pre = tf.layers.max_pooling2d(conv_pre, [2, 2], [2, 2], name="max_pool_pre")

conv_block_1 = mobile_net_block(max_pool_pre, 32, "block1")
conv_block_2 = mobile_net_block(conv_block_1, 32, "block2")
max_pool_1 = tf.layers.max_pooling2d(conv_block_2, [2, 2], [2, 2], name="max_pool_1")

conv_block_3 = mobile_net_block(max_pool_1, 32, "block3")
conv_block_4 = mobile_net_block(conv_block_3, 32, "block4")
max_pool_2 = tf.layers.max_pooling2d(conv_block_4, [2, 2], [2, 2], name="max_pool_2")


flatten = tf.layers.flatten(max_pool_2)
y_ = tf.layers.dense(flatten, 10)


loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


prob_y_ = tf.argmax(y_, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prob_y_, y), tf.float32))


# --------------- train ---------------------------------

train_data_manager = DataManager(train_file_list, need_shuffle=True)
train_data_manager.load_data()
# train_data_manager.only_use_part_data(0.2)  # full data can not run because memory is not enough

test_data_manager = DataManager(test_file_list, need_shuffle=False)
test_x, test_y = test_data_manager.load_data()
# test_data_manager.load_data()
# test_x, test_y = train_data_manager.only_use_part_data(0.2)

variable_init = tf.global_variables_initializer()

batch_size = 32
iteration = 10000

"""iteration=100000, acc=69.38, time_consume: 4015 (s)
   iteration=10000, acc=62.77, time_consume: 407 (s)
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
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
