# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:29:30 2022

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

x0 = np.array([0,0,0])



x = np.array([[1,3,1],[2,4,1],[2.5,3.5,1],[3,1.5,1],[3.5,4.5,1],[4,1.5,-1],\
              [4,3.5,-1],[4.5,2.5,-1],[5,1,-1],[6,2,-1],[6.5,3,1],[7,1.5,-1]])    
    
y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1,1,-1])




def function(w,x,y):    
    lambda_0 = 0
  
    expression = lambda_0*1/2 * np.dot(w,w) + math.log(np.prod([1+math.exp(-y[i]*np.dot(w,x[i])) for i in range(len(x))]))
    return expression

def gradient(w,x,y):   
    
    lambda_0 = 0
    expression = lambda_0*w   
        
    part1 = 1-np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))])
    
    part2 = y * part1
        
    expression = expression - x.T @ part2       
            
    return expression


def steepest_decent(x0,ita):
    iteration = 0 
    
    while iteration <100000:
        print(iteration,gradient(x0,x,y),x0)
        if np.linalg.norm(gradient(x0,x,y)) > 0.00001:
            
            dk = -ita * gradient(x0,x,y) 
            x0 = x0 +dk
            iteration = iteration + 1
            print(iteration)            
        else:        
            break
        
    return x0, iteration


if __name__ == "__main__":
    
    ita = 2

    x_opt, iteration = steepest_decent(x0,ita)








