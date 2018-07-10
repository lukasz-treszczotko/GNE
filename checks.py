# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 22:04:14 2018

@author: user
"""

import numpy as np

@np.vectorize
def f_1(x):
    return 0.
@np.vectorize
def f_2(x):
    return 10*x

def f_3(x):
    return np.sin(10*x)

def f_4(x):
    return (1/(x+0.01)) * np.sin(10*x)

test_functions = [f_1, f_2, f_3, f_4]
