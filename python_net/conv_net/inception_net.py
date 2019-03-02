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


# inception block
def inception_block(input, output_channels, name):
    with tf.variable_scope(name):
        conv1_1 = tf.layers.conv2d(input, output_channels[0], (1, 1), strides=(1, 1), padding="same",
                                 activation=tf.nn.relu, name="conv1")
        conv3_3 = tf.layers.conv2d(input, output_channels[1], (3, 3), strides=(1, 1), padding="same",
                                 activation=tf.nn.relu, name="conv2")
        conv5_5 = tf.layers.conv2d(input, output_channels[2], (5, 5), strides=(1, 1), padding="same",
                                 activation=tf.nn.relu, name="conv3")
        max_pool = tf.layers.max_pooling2d(input, [2, 2], [2, 2], name="max_pool")

    max_pool_shape = max_pool.get_shape().as_list()[1:]
    input_shape = input.get_shape().as_list()[1:]

    width_pad = (input_shape[0] - max_pool_shape[0]) // 2
    height_pad = (input_shape[1] - max_pool_shape[1]) // 2

    max_pool_pad = tf.pad(max_pool, [[0, 0], [width_pad, width_pad], [height_pad, height_pad], [0, 0]])

    output_conv = tf.concat([conv1_1, conv3_3, conv5_5, max_pool_pad], axis=3)
    return output_conv


# ------------------- net build--------------------------------
x = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name="input")
x_image = tf.reshape(x, (-1, 3, 32, 32))
x_image = tf.transpose(x_image, [0, 2, 3, 1])
y = tf.placeholder(dtype=tf.int64, shape=[None], name="y")


conv_pre = tf.layers.conv2d(x_image, 16, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv_pre")
max_pool_pre = tf.layers.max_pooling2d(conv_pre, [2, 2], [2, 2], name="max_pool_pre")

conv_block_1 = inception_block(max_pool_pre, [16, 16, 16], "block1")
conv_block_2 = inception_block(conv_block_1, [16, 16, 16], "block2")
max_pool_1 = tf.layers.max_pooling2d(conv_block_2, [2, 2], [2, 2], name="max_pool_1")

conv_block_3 = inception_block(max_pool_1, [16, 16, 16], "block3")
conv_block_4 = inception_block(conv_block_3, [16, 16, 16], "block4")
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

"""iteration=100000, acc=73.37, time_consume:  676(s)
   iteration=10000, acc=75.13, time_consume: 81 (s)
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
