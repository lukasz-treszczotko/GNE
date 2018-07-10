# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:29:20 2018

@author: 
"""
# Utils for the general VAE model 

import tensorflow as tf
import functools
from functools import partial

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

class Dense():
    """Fully-connected layer"""
    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        # (str, int, (float | tf.Tensor), tf.op)
        assert size, "Must specify layer size (num nodes)"
        self.scope = scope
        self.size = size
        self.dropout = dropout # keep_prob
        self.nonlinearity = nonlinearity
        
    def __call__(self, x):
        """Dense layer currying, to apply layer to any input tensor `x`"""
        # tf.Tensor -> tf.Tensor
        with tf.name_scope(self.scope):
            # create a name scope when called
            while True:
                try:
                    return self.nonlinearity(tf.matmul(x, self.w) + self.b)
                    # reuse weights if already initialized
                except(AttributeError):
                    self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                    # initialization to be determined later
                    self.w = tf.nn.dropout(self.w, self.dropout)
                    
    @staticmethod
    def wbVars(fan_in: int, fan_out: int):
        """Helper to initialize weights and biases, via He's adaptation
        of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
        """
        # (int, int) -> (tf.Variable, tf.Variable)
        stddev = tf.cast((2 / fan_in)**0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))
                
def composeAll(*args):
    """Util for multiple function composition
    i.e. composed = composeAll([f, g, h])
         composed(x) == f(g(h(x)))
    """
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)
    
def fully_connected(inputs, out_dim, scope_name='fc', activation=None):
    with tf.variable_scope(scope_name) as scope:
        in_dim = inputs.shape[-1]
        print(in_dim)
        w = tf.get_variable('weights', dtype=tf.float64, shape=[in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', dtype=tf.float64, shape=[out_dim],
                            initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
        if activation is not None:
            out = activation(out)
    return out
                    