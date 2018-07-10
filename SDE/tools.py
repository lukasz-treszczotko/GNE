# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 23:19:46 2018

@author: lukasz.treszczotko
"""
import numpy as np
r_0 = 0.2
from cir import CIR_v
sample_length = 100

def generate_data_CIR(samples, sample_length):
    data = CIR_v(samples, sample_length, r_0, 0.20, 0.01, 0.012, 1.)
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    return shuffled_data

data = generate_data_CIR(samples=100, sample_length=sample_length+1)
data = np.expand_dims(data, axis=2)
data_steps = data[:, 1:sample_length, :] - data[:, 0:(sample_length - 1), :]
