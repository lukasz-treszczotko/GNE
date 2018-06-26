# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:17:33 2018

@author: user
"""

# Generative noise extraction

import numpy as np
import tensorflow as tf
import kernels as ker
from fbm import FBM
import time
import matplotlib.pyplot as plt

tf.reset_default_graph()

kernel = ker.bm_kernel
num_epochs = 12
#series_length = 999
n_samples = 1000
batch_size = 10
H = 0.7
train_sample_length = 100
n = 2
sample_length = 100
state_size = 10
learning_rate = 0.02
test_size = 100
sigma = 0.1
n_batches = n_samples // batch_size
code_size = sample_length
tfd = tf.contrib.distributions

def LTB():
    import os
    os.system('tensorboard --logdir=' + 'C:/Users/user/Desktop/PYTHON/ML/Tensorflow/VAEs/graphs')
    return

def generate_data(n_samples, sample_length, H):
    """ Generates sample fBm paths using the fbm module."""
    data = np.array([FBM(n=sample_length, hurst=H, length=1,
                         method='daviesharte').fbm() for j in range(n_samples)])
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    
    return shuffled_data

def LSTM_cell(num_units, output_size):
    return tf.contrib.rnn.BasicLSTMCell(
            num_units=num_units)
      
     # reuse=tf.get_variable_scope().reuse

def GRU_cell(num_units, output_size):
    return tf.contrib.rnn.GRUCell(num_units=num_units)

def make_prior_general(code_size, kernel, time_indices):
    with tf.name_scope('prior'):
        loc = tf.zeros(code_size)
        covariance_matrix = kernel(T, T)
        scale = tf.constant(covariance_matrix)
        return tfd.MultivariateNormalDiag(loc, scale)
    
def make_wn_prior(code_size):
    with tf.name_scope('wn_prior'):
        loc = tf.zeros(code_size)
        scale =  tf.ones(code_size)
        
        return tfd.MultivariateNormalDiag(loc, scale)
    
def make_encoder(data, scope='encoder'):
    with tf.name_scope('encoder'):
        with tf.name_scope('GRU_cell_encode'):
            with tf.variable_scope('GRU_en'):
                cell_GRU_en = tf.contrib.rnn.OutputProjectionWrapper(
                        GRU_cell(num_units=state_size, output_size=n), output_size=n)
                outputs_GRU_en, states_GRU_en = tf.nn.dynamic_rnn(cell_GRU_en, data, dtype=tf.float32)
        loc = outputs_GRU_en[:,:,0]
        #print(loc.shape)
        scale =  outputs_GRU_en[:,:,1]
        scale = tf.minimum(tf.zeros_like(loc), scale)
        #scale = tf.nn.softplus(scale)
        #scale = tf.minimum(scale, tf.ones_like(scale))
        #print(scale.shape)
        return tfd.MultivariateNormalDiag(loc, scale)

def make_decoder(code, scope='decoder'):
    code = tf.expand_dims(code, axis=[2])
    with tf.name_scope('decoder'):
        with tf.name_scope('GRU_cell_decode'):
            with tf.variable_scope('GRU_de'):
                cell_GRU_de = tf.contrib.rnn.OutputProjectionWrapper(
                        GRU_cell(num_units=state_size, output_size=1), output_size=1)
                outputs_GRU_de, states_GRU_de = tf.nn.dynamic_rnn(cell_GRU_de, code, dtype=tf.float32)
        loc = outputs_GRU_de[:,:,0]
        #print(loc.shape)
        scale = 0.052*tf.ones_like(loc)
        #print(scale.shape)
        return tfd.MultivariateNormalDiag(loc, scale)
        
data = generate_data(n_samples=n_samples, sample_length=sample_length, H=H)
data = np.expand_dims(data, axis=2)


print('Version: 0.1')
print('Assembling the graph... ')
X = tf.placeholder(tf.float32, [None, sample_length + 1, 1], name='data')
#make_encoder = tf.make_template('encoder', make_encoder)
#make_decoder = tf.make_template('decoder', make_decoder)

prior = make_wn_prior(code_size=tf.shape(X)[1])
posterior = make_encoder(X)

code = posterior.sample()

#print('code shape ', code.shape)
divergence = tfd.kl_divergence(posterior, prior)
X_collapsed = tf.squeeze(X, axis=[2])
decoded = make_decoder(code)
likelihood = decoded.log_prob(X_collapsed)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(0.01).minimize(-elbo)
init = tf.global_variables_initializer()
reconstructed_version = decoded.mean()
#print('rv shape', reconstructed_version)


print('Training the network... ')
print()
with tf.Session() as sess:
    saver = tf.train.Saver()    
    start_global_time = time.time()
    sess.run(init)
    writer = tf.summary.FileWriter('graphs/', sess.graph)
    for epoch in range(num_epochs):
        for step in range(n_batches):
            X_batch = data[step * batch_size: step * batch_size + batch_size]
            #X_batch = np.squeeze(X_batch, axis=[2])
            
            
            
            _, train_elbo = sess.run([optimize, elbo], feed_dict={X: X_batch})
        if (epoch + 1) % 2 == 0:
            time_now = time.time()
            time_per_epoch = (time_now-start_global_time)/(epoch+1)
            train_elbo = sess.run([elbo], feed_dict={X: X_batch})
            print('Epoch nr: ', epoch+1, ' Current loss: ', train_elbo)
            print('Expected time remaining: %.2f seconds.' % (time_per_epoch * (num_epochs - epoch)))
            
            print(80*'_')
            
            print('Plotting a reconstructed sample...')
            choice = np.random.randint(batch_size)
            reconstruction_mean = sess.run(reconstructed_version[choice, :], feed_dict={X: X_batch})
            #reconstruction_sample = sess.run(reconstructed_sample[choice, :], feed_dict={X: X_batch})
            original_version = X_batch[choice, :]
            
            plt.figure(figsize=(11,6))
            #plt.plot(range(sample_length+1), reconstruction_sample, 'b', linewidth=1, label='Reconstructed path - samples')
            plt.plot(range(sample_length+1), reconstruction_mean, 'g', linewidth=1, label='Reconstructed path - means')
            plt.plot(range(len(original_version)), original_version, 'r', linewidth=1.4, label='Original path')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
            
            plt.show()
            print(80*'_')
            



    print('Checking for sanity...')
    x = np.linspace(0, 1, 101)
    f_x =  x - x**2
    f_x = np.expand_dims(f_x, axis=0)
    f_x = np.expand_dims(f_x, axis=2)
    reconstruction_mean = sess.run(reconstructed_version, feed_dict={X: f_x})
    original_version = f_x
    plt.plot(range(sample_length+1), reconstruction_mean[0], 'g', linewidth=1, label='Reconstructed path - means')
    plt.plot(range(sample_length+1), original_version[0], 'r', linewidth=1.4, label='Original path')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
            
    plt.show()
    












