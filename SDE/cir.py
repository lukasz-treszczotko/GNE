# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:59:22 2018

@author: lukasz.treszczotko
"""
import numpy as np
import matplotlib.pyplot as plt

def CIR(r0, K, theta, sigma, T=1., N=100):
    
    dt = T/float(N)    
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + \
            sigma*np.sqrt(rates[-1])*np.random.normal()
        rates.append(rates[-1] + dr)
    return range(N+1), rates

def test_CIR():
    x, y = CIR(0.01875, 0.20, 0.01, 0.012, 1., 1000)
    plt.plot(x,y)
    plt.show()
    


def CIR_v(samples, sample_length, r0, K, theta, sigma, T=1.):
    dt = T/float(sample_length) 
    
    rates = np.zeros([samples, sample_length + 1], dtype=np.float32)
    rates[:, 0] = r0
    for i in range(sample_length):
        dr = K*(theta-rates[:, i])*dt + \
            sigma*np.sqrt(rates[:, i])*np.random.normal(size=samples)
        rates[:, i+1] = rates[:, i] + dr
    return rates

def test_CIR_v():
    x = np.linspace(0, 1, 101)
    y = CIR_v(100, 100, 0.01875, 0.20, 0.01, 0.012, 1.)
    plt.plot(x,y[0])
    plt.show()
    
if __name__ == '__main__':
    test_CIR()
    test_CIR_v()
        
    
    