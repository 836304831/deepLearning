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


# res_net block
def res_bottleneck_design_block(input, output_channel):
    input_channel = input.get_shape()[-1]
    conv1 = tf.layers.conv2d(input, output_channel, (1, 1), strides=(1, 1), padding="same",
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, output_channel, (3, 3), strides=(1, 1), padding="same",
                             activation=tf.nn.relu, name="conv2")
    conv3 = tf.layers.conv2d(conv2, output_channel * 4, (1, 1), strides=(1, 1), padding="same",
                             activation=tf.nn.relu, name="conv3")

    res_channel = output_channel * 4 - input_channel
    print("res_channel:{}".format(res_channel))
    print(input.shape)
    padding_x = tf.pad(input, [[0, 0], [0, 0], [0, 0], [res_channel // 2, res_channel // 2]])

    print(padding_x.shape, conv3.shape)
    outpnut = padding_x + conv3
    max_pool = tf.layers.max_pooling2d(outpnut, [2, 2], strides=[1, 1])
    return max_pool


# ------------------- net build--------------------------------
x = tf.placeholder(dtype=tf.float32, shape=[None, 3072], name="input")
x_image = tf.reshape(x, (-1, 3, 32, 32))
x_image = tf.transpose(x_image, [0, 2, 3, 1])
y = tf.placeholder(dtype=tf.int64, shape=[None], name="y")

layers = []
num_block_channel = [16, 32]
with tf.variable_scope("pre_conv"):
    conv_pre = tf.layers.conv2d(x_image, 16, [3, 3], [1, 1], padding="same", activation=tf.nn.relu, name="conv1_1")
    layers.append(conv_pre)

for index in range(len(num_block_channel)):
    with tf.variable_scope("block%d" % index):
        conv = res_bottleneck_design_block(layers[-1], num_block_channel[index])
        layers.append(conv)

with tf.variable_scope("fc"):
    global_pool = tf.reduce_mean(layers[-1], [1, 2])
    output = tf.layers.dense(global_pool, 10)

y_ = output
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
train_data_manager.only_use_part_data(0.2)

test_data_manager = DataManager(test_file_list, need_shuffle=False)
# test_x, test_y = test_data_manager.load_data()
test_data_manager.load_data()
test_x, test_y = train_data_manager.only_use_part_data(0.2)

variable_init = tf.global_variables_initializer()

batch_size = 20
iteration = 100000

"""iteration=100000, acc=100.0, time_consume: 675 (s)
   iteration=10000, acc=67.75, time_consume: 68 (s)
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
