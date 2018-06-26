# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:23:53 2018

@author: user
"""

# Gaussian Processes Kernels 

import numpy as np

def exponential_cov(x, y, params):
    return params[0] * np.exp(-0.5 * params[1] *
                 np.subtract.outer(x, y)**2)
    # returns a covarince matrix based on an exponenstial 
    
def fbm_kernel(x, y, H):
    #x = np.squeeze(x)
    #x = np.squeeze(y)
    def op(a, b, h):
        return (1/2.) * (np.abs(a)**(2*h) + np.abs(b)**(2*h) - np.abs(a-b)**(2*h))
    r = np.zeros((len(x),len(y)))
    
    for i in range(len(x)):
        for j in range(len(y)):
            r[i,j] = op(x[i], y[j], H) # op = ufunc in question    
    return r
    
    # covariance kernel
    
def bm_kernel(x, y):
    #x = np.squeeze(x)
    #x = np.squeeze(y)
    def op(a, b):
        return np.min(a, b)
    r = np.zeros((len(x),len(y)))
    
    for i in range(len(x)):
        for j in range(len(y)):
            r[i,j] = op(x[i], y[j]) # op = ufunc in question    
    return r
    
def conditional_exp(x_new, x, y, params):
    # define covariance matrices
    B = exponential_cov(x_new, x, params)
    C = exponential_cov(x, x, params)
    A = exponential_cov(x_new, x_new, params)
    
    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T))
    
    return (mu.squeeze(), sigma.squeeze())