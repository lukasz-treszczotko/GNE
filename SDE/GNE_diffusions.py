# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 20:03:54 2018

@author: lukasz.treszczotko

A general model for noise extraction given a diffusion.
"""

import numpy as np
import tensorflow as tf
from cir import CIR_v
tfd = tf.contrib.distributions


import time
import matplotlib.pyplot as plt


version = 0.3
tf.reset_default_graph()
r_0 = 0.18
version = 0.3
num_epochs = 20
#series_length = 999
n_samples = 500
batch_size = 32
sample_length = 100
state_size = 10
n = 1
n_batches = n_samples // batch_size
code_size = sample_length-1
num_outputs = 2
activation = tf.nn.tanh
tfd = tf.contrib.distributions

hidden_1 = 200
hidden_2 = 200

def generate_data_CIR(samples, sample_length):
    data = CIR_v(samples, sample_length, r_0, 0.20, 0.01, 0.012, 1.)
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    return shuffled_data

data = generate_data_CIR(samples=n_samples, sample_length=sample_length)

X = tf.placeholder(tf.float32, [batch_size, sample_length], name='X')
W = tf.placeholder(tf.float32, [1,1], name='W')




with tf.variable_scope("weights", reuse=tf.AUTO_REUSE):
        
    w_1 = tf.get_variable('w_1', shape=[1, hidden_1], dtype=tf.float32, 
            initializer=tf.truncated_normal_initializer())
    b_1 = tf.get_variable('b_1', shape=[hidden_1],
                          dtype=tf.float32, 
                          initializer=tf.truncated_normal_initializer())
    dense_1 = activation(tf.matmul(W, w_1) + b_1)
    w_2 = tf.get_variable('w_2', shape=[hidden_1, hidden_2],
                          dtype=tf.float32, 
                          initializer=tf.truncated_normal_initializer())
    b_2 = tf.get_variable('b_2', shape=[hidden_2],
                          dtype=tf.float32, 
                          initializer=None)
    dense_2 = activation(tf.matmul(dense_1, w_2) + b_2)
    w_3 = tf.get_variable('w_3', shape=[hidden_2, num_outputs],
                          dtype=tf.float32, 
                          initializer=tf.truncated_normal_initializer())
    b_3 = tf.get_variable('b_3', shape=[num_outputs],
                          dtype=tf.float32, 
                          initializer=None)
    outputs = tf.matmul(dense_2, w_3) + b_3


 
init = tf.global_variables_initializer()

def test_param_computation():
    test_value = 1.
    test_value = np.reshape(test_value, [1,1])
    #test_value = np.expand_dims(test_value, axis=0)
    #
    test_value = np.expand_dims(test_value, axis=1)
    with tf.Session() as sess:
        sess.run(init)
        result = sess.run([outputs], feed_dict={W: test_value})
        print(result)

    
    
if __name__ == '__main__':
    test_param_computation()
