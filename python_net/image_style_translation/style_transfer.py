# -*- coding: utf-8 -*-
# -----------------------------------
# File: style_transfer.py
# Author: acedar
# Date: 2019.03.21
# -----------------------------------

import os
import math
import tensorflow as tf
import numpy as np
import time
from PIL import Image


VGG_MEAN = [103.939, 116.779, 123.68]  # [b, g , r]


class VggNet(object):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="conv")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="fc")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="bias")

    def conv_layer(self, input, name):
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            bias_b = self.get_bias(name)
            h = tf.nn.conv2d(input, conv_w, [1, 1, 1, 1], padding="SAME")
            h = tf.nn.bias_add(h, bias_b)
            h = tf.nn.relu(h)
            return h

    def pool_layer(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)

    def fc_layer(self, x, name, activation=tf.nn.relu):
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            if activation is None:
                return h
            h = tf.nn.relu(h)
            return h

    def flatten_layer(self, x, name):
        with tf.name_scope(name):
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            h = tf.reshape(x, [-1, dim])
            return h

    def build(self, x_rgb):
        start_build_time = time.time()

        r_channel, g_channel, b_channel = tf.split(x_rgb, [1, 1, 1], axis=3)
        bgr = tf.concat([tf.cast(b_channel, tf.float32) - VGG_MEAN[0],
                         tf.cast(g_channel, tf.float32) - VGG_MEAN[1],
                         tf.cast(r_channel, tf.float32) - VGG_MEAN[2]],
                        axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, name="conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, name="conv1_2")
        self.pool1 = self.pool_layer(self.conv1_2, name="pool1")

        self.conv2_1 = self.conv_layer(self.pool1, name="conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, name="conv2_2")
        self.pool2 = self.pool_layer(self.conv2_2, name="pool2")

        self.conv3_1 = self.conv_layer(self.pool2, name="conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, name="conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, name="conv3_3")
        self.pool3 = self.pool_layer(self.conv3_3, name="pool3")

        self.conv4_1 = self.conv_layer(self.pool3, name="conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, name="conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, name="conv4_3")
        self.pool4 = self.pool_layer(self.conv4_3, name="pool4")

        self.conv5_1 = self.conv_layer(self.pool4, name="conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, name="conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, name="conv5_3")
        self.pool5 = self.pool_layer(self.conv5_3, name="pool5")

        self.flatten = tf.layers.flatten(self.pool5, name="flatten")
        """
        self.fc6 = self.fc_layer(self.flatten, name="fc6")
        self.fc7 = self.fc_layer(self.fc6, name="fc7")
        self.fc8 = self.fc_layer(self.fc7, name="fc8", activation=None)
        
        self.prob = tf.nn.softmax(self.fc8)
        """
        print("build net time: {} s".format(time.time() - start_build_time))


def initial_result(shape, mean, stddev):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def read_img(img_name, width=224, height=224):
    img = Image.open(img_name)
    img = img.resize((width, height), Image.ANTIALIAS)
    np_img = np.array(img)

    # change shape=3 to shape=4
    np_img = np.asarray([np_img], dtype=np.int32)
    return np_img


def gram_matrix(x):
    """caculate gram matrix
      Args:
      - x: features extracted from vgg net, shape: [1, width, heigth, channel]
    """
    b, w, h, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h*w, ch])
    # [h*w, ch] matrix -> [ch, h*w] * [h*w, ch] -> [ch, ch]
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w *h, tf.float32)
    return gram


__dir__ = os.path.abspath(os.path.dirname(__file__))
WORK_DIR = os.path.dirname(os.path.dirname(__dir__))
print(WORK_DIR)
vgg16_model_path = os.path.join(WORK_DIR, "datasets/trained_model/vgg16.npy")
content_img_path = os.path.join(WORK_DIR, "datasets/images/gugong.jpg")
style_img_path = os.path.join(WORK_DIR, "datasets/images/fengjing.png")

num_steps = 100
learning_rate = 0.1

lambda_c = 0.1
lambda_s = 500

output_dir = os.path.join(WORK_DIR, "datasets/style_transfer_images")


result = initial_result((1, 224, 224, 3), 127.5, 10)

print(os.path.exists(content_img_path))
print(os.path.exists(style_img_path))
content_val = read_img(content_img_path)
# content_val = content_val.resize((224, 224))

style_val = read_img(style_img_path)
style_val = style_val.transpose([3, 0, 1, 2])
print("********convert 4 channel to 3 channel*******")
style_val = np.asarray(style_val[0: 3])
style_val = style_val.transpose([1, 2, 3, 0])

content_placeholder = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
style_placeholder = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

vgg16_model_dict = np.load(vgg16_model_path, encoding='latin1').item()
vgg_for_content = VggNet(vgg16_model_dict)
vgg_for_style = VggNet(vgg16_model_dict)
vgg_for_result = VggNet(vgg16_model_dict)

print(content_val.shape)
print(style_val.shape)
print(result.shape)
vgg_for_content.build(content_val)
vgg_for_style.build(style_val)
vgg_for_result.build(result)

content_features = [
    vgg_for_content.conv1_2,
    # vgg_for_content.conv2_2,
    # vgg_for_content.conv3_3,
    # vgg_for_content.conv4_3,
    # vgg_for_content.conv5_3
]
result_content_features = [
    vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    # vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]

style_features = [
    # vgg_for_style.conv1_2,
    # vgg_for_style.conv2_2,
    # vgg_for_style.conv3_3,
    vgg_for_style.conv4_3,
    # vgg_for_style.conv5_3
]

result_style_features = [
    # vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3
]

content_loss = tf.zeros(1, tf.float32)
for c, c_ in zip(content_features, result_content_features):
    content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])

style_gram = [gram_matrix(feature) for feature in style_features]
result_style_gram = [gram_matrix(feature) for feature in result_style_features]

style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    style_loss += tf.reduce_mean((s - s_) **2, [1, 2])

loss = content_loss * lambda_c + style_loss * lambda_s
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        loss_val, content_loss_val, style_loss_val, _ = sess.run([loss, content_loss, style_loss, train_op],
                                                                 feed_dict={content_placeholder: content_val,
                                                                            style_placeholder: style_val})
        if (step + 1) % 5 == 0:
            print("step: %d, loss_val: %8.4f, content_loss: %8.4f, style_loss: %8.4f"
                  % ((step + 1), loss_val[0], content_loss_val[0], style_loss_val[0]))
            result_img_path = os.path.join(output_dir, "result-%5d.jpg" % (step + 1))
            result_val = result.eval(sess)[0]
            result_val = np.clip(result_val, 0, 255)
            img_arr = np.asarray(result_val, np.uint8)
            img = Image.fromarray(img_arr)
            img.save(result_img_path)
