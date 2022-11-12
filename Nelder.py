# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:59:47 2022

@author: amit
"""


import scipy
import scipy.linalg

import numpy as np
import sympy as sym
from sympy import *
from sympy import lambdify
from matplotlib import pyplot as plt
import math
import time
from scipy.optimize import minimize

# Nelder mead method


x0 = np.array([0,0,0])

    
x = np.array([[1,3,0],[2,4,0],[2.5,3.5,0],[3,1.5,0],[3.5,4.5,0],[4,1.5,0],\
              [4,3.5,0],[4.5,2.5,0],[5,1,0],[6,2,0],[6.5,3,0],[7,1.5,0]])
    
y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1,1,-1])


def function(w,x,y):    
    lambda_0 = 0
    expression = lambda_0 * (1/2) * np.dot(w,w)
    for i in range(len(y)):
        expression =  expression + math.log(1+math.exp(-y[i]*np.dot(w,x[i])))
                
    return expression


    

def nelder_mead(x0,alpha,beta,gamma,epsilon,x,y):

    fun = [function(x[i],x,y) for i in range(len(x))]
    
    err_sum = sum([abs(fun[i]-function(x0,x,y))**2 for i in range(len(fun))])
    iteration = 0
    #reflection
    
    while err_sum > epsilon:
    
        sorting_tuple = list(zip(fun,list(x),list(y)))
        
        sorting_tuple.sort(key=lambda tup: tup[0])
        
        fun = np.array([sorting_tuple[i][0] for i in range(len(sorting_tuple))])    
        
        x = np.array([sorting_tuple[i][1] for i in range(len(sorting_tuple))])
        
        y = np.array([sorting_tuple[i][2] for i in range(len(sorting_tuple))])
        
        
        xr = x0 + alpha*(x0-x[-1])
        fr = function(xr,x,y)
        if fr >= fun[0] and fr < fun[-2]:
            x[-1] = xr
            
        elif fr < fun[0]:
            xe = x0 + beta*(xr-x0)
            if function(xe,x,y) < fr:
                x[-1] = xe
            else:
                x[-1] = xr
                
        elif fr >= fun[-2]:
            if fr >= fun[-1]:
                xk = x0 + gamma*(x[-1] - x0)
                if function(xk,x,y) < fun[-1]:
                    x[-1] = xk
                else:
                    x = np.array([(x[i] + x[0])/2 for i in range(len(y))])
            else:
                xk = x0 + gamma*(xr - x0)
                if function(xk,x,y) < fr:
                    x[-1] = xk
                else:
                    x = np.array([(x[i] + x[0])/2 for i in range(len(y))])
                    
        fun = np.array([function(x[i],x,y) for i in range(len(y))])        
        
        x0 = sum(x)/len(x) # average of x
        
        
                
        err_sum = sum([abs(fun[i]-function(x0,x,y))**2 for i in range(len(fun))])/len(x)
        print(err_sum)
        
        
        if err_sum < epsilon:
            break
        else:
            iteration = iteration + 1
            print(iteration)
    return x , x0, fun, iteration

if __name__ == "__main__":    
    alpha = 0.5
    beta = 1.5
    gamma = 0.5
    epsilon = 0.000001
    
    x1 , x01, fun1, iteration1 = nelder_mead(x0,alpha,beta,gamma,epsilon,x,y)

