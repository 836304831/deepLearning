# -*- coding: utf-8 -*-
# -----------------------------------
# File: vgg16_model.py
# Author: acedar
# Date: 2019.03.18
# -----------------------------------

import os
import pickle
import numpy as np

__dir__ = os.path.abspath(os.path.dirname(__file__))

WORK_DIR = os.path.dirname(__dir__)
vgg16_model_path = os.path.join(WORK_DIR, "datasets/trained_model/vgg16.npy")

print(os.path.exists(vgg16_model_path))
vgg16_model = np.load(vgg16_model_path, encoding='latin1').item()

print("type: {}".format(type(vgg16_model)))

print("layer keys size: {}".format(len(vgg16_model)))

print("layer keys : {}".format(vgg16_model.keys()))

print("conv parse......")
conv1_1 = vgg16_model.get("conv1_1")
print("conv type: {}".format(type(conv1_1)))
print("conv shape: {}".format(len(conv1_1)))
w, b = conv1_1
print("w.shape: {}".format(w.shape))
print("b.shape: {}".format(b.shape))

print("fc6 parse......")
fc6 = vgg16_model["fc6"]
print("fc6 len: {}".format(len(fc6)))
w, b = fc6
print("w.shape: {}".format(w.shape))
print("b.shape: {}".format(b.shape))
