# -*- coding: utf-8 -*-
# -----------------------------------
# File: binary_classification_net.py
# Author: acedar
# Date: 2019.03.11
# -----------------------------------

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

__dir__ = os.path.abspath(os.path.dirname(__file__))
WORK_DIR = os.path.dirname(os.path.dirname(__dir__))


img_path = os.path.join(WORK_DIR, "datasets/images/fengjing.png")
print("img_path exist: {}, img_path: {}".format(os.path.exists(img_path), img_path))

# read image with tensorflow
img_str = tf.read_file(img_path)
img_decoded = tf.image.decode_image(img_str)
sess = tf.Session()
img_decoded_value = sess.run(img_decoded)
print("img_type: {}, {}".format(type(img_decoded_value), img_decoded_value.shape))



# resize
# tf.image.resize_area
# tf.image.resize_bicubic
# tf.image.resize_bilinear
# tf.image.resize_nearest_neighbor
reshape_img = tf.reshape(img_decoded, [1, img_decoded_value.shape[0], img_decoded_value.shape[1], img_decoded_value.shape[2]])
resized_img = tf.image.resize_area(reshape_img, [700, 1200])
img_resized_img_value = sess.run(resized_img)
reshaped_img = img_resized_img_value.reshape((700, 1200, 4))
reshaped_img = np.asarray(reshaped_img, np.uint8)
plt.imshow(reshaped_img)
plt.show()


# crop
# tf.image_pad_to_bounding_box
# tf.image.crop_to_bounding_box
# tf.image.random_crop
reshape_img = tf.reshape(img_decoded, [1, img_decoded_value.shape[0], img_decoded_value.shape[1], img_decoded_value.shape[2]])
# padded_img = tf.image.pad_to_bounding_box(reshape_img, 50, 100, 600, 800)
# padded_img_value = sess.run(padded_img)

croped_img = tf.image.crop_to_bounding_box(reshape_img, 50, 100, 300, 400)
croped_img_value = sess.run(croped_img)

reshaped_img = croped_img_value.reshape((300, 400, 4))
reshaped_img = np.asarray(reshaped_img, np.uint8)
plt.imshow(reshaped_img)
plt.show()


# flip
# tf.image.flip_left_right
# tf.image.flip_up_down
# tf.image.random_flip_left_right
# tf.image.random_flip_up_down

reshape_img = tf.reshape(img_decoded, [1, img_decoded_value.shape[0], img_decoded_value.shape[1],
                         img_decoded_value.shape[2]])
# flipped_img = tf.image.flip_up_down(reshape_img)
flipped_img = tf.image.flip_left_right(reshape_img)

flipped_img_value = sess.run(flipped_img)
flipped_img_value = flipped_img_value.reshape((flipped_img_value.shape[1], flipped_img_value.shape[2],
                                               flipped_img_value.shape[3]))
flipped_img_value = np.asarray(flipped_img_value, np.uint8)
print(flipped_img_value.shape)
plt.imshow(flipped_img_value)
plt.show()


# brightness/contrast
# tf.image.adjust_brightness
# tf.image.random_brightness
# tf.image.adjust_contrast
# tf.image.random_contrast

reshape_img = tf.reshape(img_decoded, [1, img_decoded_value.shape[0], img_decoded_value.shape[1],
                         img_decoded_value.shape[2]])
# brighted_img = tf.image.adjust_brightness(reshape_img, -0.5)
# brighted_img = tf.image.adjust_brightness(reshape_img, 0.5)
brighted_img = tf.image.adjust_contrast(reshape_img, 0.5)
brighted_img_value = sess.run(brighted_img)
brighted_img_value = np.reshape(brighted_img_value, (brighted_img_value.shape[1], brighted_img_value.shape[2],
                                 brighted_img_value.shape[3]))
brighted_img_value = np.asarray(brighted_img_value, np.uint8)
print(brighted_img_value.shape)
plt.imshow(brighted_img_value)
plt.show()
