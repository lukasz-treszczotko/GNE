# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 18:56:38 2018

@author: lukasz.treszczotko
"""

import numpy as np
import tensorflow as tf
from cir import CIR_v
tfd = tf.contrib.distributions
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import time



version = 0.4
tf.reset_default_graph()
r_0 = 0.18

num_epochs = 20
#series_length = 999
n_samples = 2000
batch_size = 100
sample_length = 100
state_size = 10
n = 1
n_batches = n_samples // batch_size
code_size = sample_length
num_outputs = 2

timespan = 1.
timestep = timespan/sample_length



activation = tf.nn.relu
tfd = tf.contrib.distributions

hidden_1 = 200
hidden_2 = 200

def generate_data_CIR(samples, sample_length):
    data = CIR_v(samples, sample_length, r_0, 0.20, 0.01, 0.012, 1.)
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    return shuffled_data

data = generate_data_CIR(samples=n_samples, sample_length=sample_length)

print('Version: ', version)
print('Assembling the graph... ')

X = tf.placeholder(tf.float32, [batch_size, sample_length+1], name='X')


X_reshaped = tf.reshape(X, [-1, 1])






with tf.variable_scope("weights"):
        
    w_1 = tf.get_variable('w_1', shape=[1, hidden_1], dtype=tf.float32, 
            initializer=None)
    b_1 = tf.get_variable('b_1', shape=[hidden_1],
                          dtype=tf.float32, 
                          initializer=None)
    dense_1 = activation(tf.matmul(X_reshaped, w_1) + b_1)
    
    w_2 = tf.get_variable('w_2', shape=[hidden_1, hidden_2],
                          dtype=tf.float32, 
                          initializer=None)
    b_2 = tf.get_variable('b_2', shape=[hidden_2],
                          dtype=tf.float32, 
                          initializer=None)
    dense_2 = activation(tf.matmul(dense_1, w_2) + b_2)
    w_3 = tf.get_variable('w_3', shape=[hidden_2, num_outputs],
                          dtype=tf.float32, 
                          initializer=None)
    b_3 = tf.get_variable('b_3', shape=[num_outputs],
                          dtype=tf.float32, 
                          initializer=None)
    outputs = tf.matmul(dense_2, w_3) + b_3
    #utputs = tf.nn.softplus(outputs)
    init = tf.global_variables_initializer()

#X_params = outputs
#print(X_params.shape)
#X_params = tf.reshape(X_params, [batch_size, sample_length,2])
#print(X_params.shape)
#X_params_b = X_params[:, :-1, 0]
#print(X_params_b.shape)
#X_params_sigma = X_params[:, :-1, 1]
#print(X_params_sigma.shape)

#driftless_noise = X_steps - timestep * X_params_b
#noise = tf.divide(driftless_noise, X_params_sigma)

def compute_params(W):
    """ W must be a two dimensional tensor."""
    shape = W.get_shape().as_list()
    
    shape.append(2)
   
    
    W = tf.reshape(W, [-1,1])
    dense_1 = activation(tf.matmul(W, w_1) + b_1)
    dense_2 = activation(tf.matmul(dense_1, w_2) + b_2)
    outputs = tf.matmul(dense_2, w_3) + b_3
    #outputs = tf.nn.softplus(outputs)
    return tf.reshape(outputs,shape)

X_params = compute_params(X)

    
def make_wn_prior(code_size):
    with tf.name_scope('wn_prior'):
        loc = tf.zeros(code_size)
        scale =  timestep * tf.ones(code_size)
        
    return tfd.MultivariateNormalDiag(loc, scale)

def make_encoder(X, scope='encoder'):
    with tf.name_scope('encoder'):
        X_steps = X[:, 1:sample_length+1] - X[:, 0:(sample_length )]
        X_params = compute_params(X)
        print('hej', X_params.shape)
        
        X_params = X_params[:, :-1, :]
        
        driftless_noise = X_steps - timestep * X_params[:,:,0]
        #variance = X_params[:, :,1]
        noise = tf.divide(driftless_noise, X_params[:, :,1])
        
        
        loc = driftless_noise
        
        scale = noise
        #scale = tf.minimum(scale, tf.ones_like(scale))
        #print(scale.shape)
    return tfd.MultivariateNormalDiag(loc, scale)
    
def make_decoder(code, scope='decoder'):
    
    code= tf.transpose(code, [1,0])
    
    
    start = X[:, 0]
    
 
    def fn(prev, current):
        result = prev + timestep*compute_params(prev)[:, 0] \
            + tf.multiply(compute_params(prev)[:, 1], current) 
        result = tf.maximum(result, 0.001)
        return result
    output = tf.scan(fn=fn, elems=code, initializer=start)
    start = tf.expand_dims(start, axis=0)
    conc = tf.concat([start, output], 0)
    conc = tf.transpose(conc, [1, 0])
    
    loc = conc
    loc = tf.maximum(loc, 0.001)
    scale = (0.01) * tf.ones_like(conc) / timestep
    return tfd.MultivariateNormalDiag(loc, scale)
    

prior = make_wn_prior(code_size=sample_length)
posterior = make_encoder(X)

code = posterior.sample()


divergence = tfd.kl_divergence(posterior, prior)
#X_collapsed = tf.squeeze(X, axis=[2])
decoded = make_decoder(code)
likelihood = decoded.log_prob(X)
elbo = tf.reduce_mean(likelihood - divergence)

optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)
init = tf.global_variables_initializer()
reconstructed_version = decoded.mean()


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
            

    saver.save(sess, "./version_" + str(version))

    print('Checking for sanity...')
    x = np.linspace(0, 1, sample_length)
    f_x =  x - x**2*np.sin(x)
    f_x = np.expand_dims(f_x, axis=0)
    f_x = np.expand_dims(f_x, axis=2)
    reconstruction_mean = sess.run(reconstructed_version, feed_dict={X: f_x})
    original_version = f_x
    plt.plot(range(sample_length+1), reconstruction_mean[0], 'g', linewidth=1, label='Reconstructed path - means')
    plt.plot(range(sample_length+1), original_version[0], 'r', linewidth=1.4, label='Original path')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    plt.show()
    

    

    

     

        

 


