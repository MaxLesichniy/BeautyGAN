# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import numpy as np
import os
import glob
from imageio.v2 import imread, imsave
import cv2
import argparse

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--ref', type=str)
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

input_file_name = os.path.basename(args.input)
input_file_name = os.path.splitext(input_file_name)[0]

batch_size = 1
img_size = 256
input = cv2.resize(imread(args.input), (img_size, img_size))
ref = cv2.resize(imread(args.ref), (img_size, img_size))
X_img = np.expand_dims(preprocess(input), 0)
Y_img = np.expand_dims(preprocess(ref), 0)

ops.reset_default_graph()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join(args.model, 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint(args.model))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
Xs_ = deprocess(Xs_)

imsave(args.output, ( Xs_[0] * 255).astype(np.uint8))

# test image grid
makeups = glob.glob(os.path.join('/Users/maxlesichniy/MuzaServerData/BeautyGAN/imgs', 'makeup', '*.*'))
comb_result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
comb_result[img_size: 2 *  img_size, :img_size] = input / 255.

for i in range(len(makeups)):
    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    Xs_ = deprocess(Xs_)
    comb_result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
    comb_result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]
    
comb_result_path = os.path.split(args.output)[0] + "/" + input_file_name + "_comb_result.jpg"
imsave(comb_result_path, (comb_result * 255).astype(np.uint8))