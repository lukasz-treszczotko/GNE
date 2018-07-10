# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:27:04 2018

@author: user
"""

# A general autoencoder model that can be easily tuned to serve different
# purposes



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import Dense

from utils import fully_connected

from utils import composeAll

from fbm import FBM
import time
tfd = tf.contrib.distributions

# Hyperparamters:
version = 1
n_samples = 4000
sample_length = 100
H = 0.7
code_size = 4
hidden_size = 200
noise=0.05
num_epochs = 20
batch_size = 100
n_batches = n_samples // batch_size



tf.reset_default_graph()

def generate_data(n_samples, sample_length, H):
    """ Generates sample fBm paths using the fbm module."""
    data = np.array([FBM(n=sample_length, hurst=H, length=1,
                         method='daviesharte').fbm() for j in range(n_samples)])
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    
    return shuffled_data
    
def generate_labeled_data(n_samples_each, sample_length, H_1, H_2):
    """ Generates sample fBm paths using the fbm module."""
    data_1 = np.array([FBM(n=sample_length, hurst=H_1, length=1,
                         method='daviesharte').fbm() for j in range(n_samples)])
    permutation_1 = np.random.permutation(data_1.shape[0])
    data_1 = data_1[permutation_1]
    
    data_2 = np.array([FBM(n=sample_length, hurst=H_2, length=1,
                         method='daviesharte').fbm() for j in range(n_samples)])
    permutation_2 = np.random.permutation(data_2.shape[0])
    data_2 = data_2[permutation_2]
    
    data_1 = np.pad(data_1, ((0,0), (0,1)), mode='constant', constant_values=H_1)
    data_2 = np.pad(data_1, ((0,0), (0,1)), mode='constant', constant_values=H_2)
    return data_1, data_2
    
    
    
    
    
    
    
    
data = generate_data(n_samples, sample_length, H)
# a numpy array
#dataset = tf.contrib.data.Dataset.from_tensor_slices(data)
#print(dataset.output_shapes)
#dataset.batch(batch_size)
#print(dataset.output_shapes)
#print(dataset.output_types)


def make_encoder(data, code_size, scope='encoder'):
    with tf.name_scope('encoder'):
        h_1 = tf.layers.dense(data, hidden_size, activation=tf.nn.relu, name='hidden_1')
        h_2 = tf.layers.dense(h_1, hidden_size, tf.nn.relu, name='hidden_2')
        loc = tf.layers.dense(h_2, code_size, name='loc')
        scale = tf.layers.dense(h_2, code_size, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)
  
def make_prior(code_size):
    with tf.name_scope('prior'):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        return tfd.MultivariateNormalDiag(loc, scale)

def make_decoder(code, data_shape):
    with tf.name_scope('decoder'):
        
        h_1 = tf.layers.dense(code, hidden_size, tf.nn.relu, name='hidden_1')
        h_2 = tf.layers.dense(h_1, hidden_size, tf.nn.relu, name='hidden_2')
        
        loc = tf.layers.dense(h_2, np.prod(data_shape), name='loc')
        scale = tf.layers.dense(h_2, np.prod(data_shape), 
                                activation=tf.nn.relu, 
                                name='scale',
                                use_bias=False)
        scale = tf.add(tf.nn.softplus(scale), noise)
        return tfd.MultivariateNormalDiag(loc=loc,
                                          scale_diag=scale)
                                          
                                          #0.05*tf.ones_like(loc))
                                          
                                          
                                          
        
X = tf.placeholder(tf.float32, [None, sample_length + 1], name='data')
make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

prior = make_prior(code_size=code_size)
posterior = make_encoder(X, code_size=code_size)
code = posterior.sample()

# 2. Define the loss
reconstructed_version = make_decoder(code, [sample_length + 1]).mean()
reconstructed_sample = make_decoder(code, [sample_length + 1]).sample()
likelihood = make_decoder(code, [sample_length + 1]).log_prob(X)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

samples = make_decoder(prior.sample(10), [sample_length + 1]).mean()


    


init = tf.global_variables_initializer()




print('Version: %d' % (version))
print('Training the network... ')
print()
with tf.Session() as sess:
    saver = tf.train.Saver()    
    start_global_time = time.time()
    sess.run(init)
    writer = tf.summary.FileWriter('graphs/alpha_version', sess.graph)
    for epoch in range(num_epochs):
        for step in range(n_batches):
            X_batch = data[step * batch_size: step * batch_size + batch_size, :]
            _, train_elbo = sess.run([optimize, elbo], feed_dict=
                                        {X: X_batch})
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
            plt.figure(figsize=(14,8))
            #plt.plot(range(sample_length+1), reconstruction_sample, 'b', linewidth=1, label='Reconstructed path - samples')
            plt.plot(range(sample_length+1), reconstruction_mean, 'g', linewidth=1, label='Reconstructed path - means')
            plt.plot(range(len(original_version)), original_version, 'r', linewidth=1.4, label='Original path')
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()
            print(80*'_')
            
            print('Plotting the code on the latent_space...')
            choice2 = np.random.randint(n_batches)
            z = sess.run(posterior.sample(), feed_dict=
                {X:data[choice2 * batch_size: choice2 * batch_size + batch_size, :]})
            
            
            plt.figure(figsize=(8,8))
            colors = (0,0,0)
            plt.scatter(z[:, 0], z[:, 1], c=colors, alpha=0.5)
            plt.show()
            
            
        
            
        saver.save(sess, "./version_" + str(version))
        writer.close()
    samples_to_plot = sess.run(samples)
    from checks import test_functions
    x = np.linspace(0, 1, sample_length+1)
    for j in range(4):
        values = test_functions[j](x)
        values = np.expand_dims(values, axis=0)
        
        reconstruction_mean = sess.run(reconstructed_version, 
                                       feed_dict={X: values})
        plt.plot(range(sample_length+1), reconstruction_mean[0], 'g', linewidth=1, label='Reconstructed path - means')
        plt.plot(range(sample_length+1), values[0], 'r', linewidth=1.4, label='Original path')
        plt.legend()
        plt.show()

#def plot_r_sample():
#for j in range(10):
    #plt.figure(figsize=(14,8))
    #sample_vector = samples_to_plot[j, :]
    #plt.plot(range(len(sample_vector)), sample_vector, 'b', linewidth=1, 
             #label='Generated path')
    #plt.xlim([0,sample_length])
    #plt.legend(loc=0)
    #plt.title('Error = %.5f' % (predicted_values - sample_vector[-1]))
    #plt.show()
    

             
def sample_generator():
    with tf.Session() as sess:
        
        code_sample = sess.run(prior.sample())
        saver.restore(sess, "./version_" + str(version))
        code_sample = np.expand_dims(code_sample, axis=0)
        generated_sample = sess.run(reconstructed_version, 
                                    feed_dict={code: code_sample})
    return np.squeeze(generated_sample)
    
def plot_sample(sample):
    
    plt.figure(figsize=(14,8))
    plt.plot(range(sample_length+1), sample, 'g', linewidth=2, 
             label='Generated_path, H =' + str(H))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()