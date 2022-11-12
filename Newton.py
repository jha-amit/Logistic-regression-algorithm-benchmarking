# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 20:07:21 2022

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
    lambda_0 = 1
  
    expression = lambda_0*1/2 * np.dot(w,w) + math.log(np.prod([1+math.exp(-y[i]*np.dot(w,x[i])) for i in range(len(x))]))
    return expression

def function(w,x,y):    
    lambda_0 = 0
    expression = lambda_0 * (1/2) * np.dot(w,w)
    for i in range(len(y)):
        expression =  expression + math.log(1+math.exp(-y[i]*np.dot(w,x[i])))
                
    return expression

def gradient(w,x,y):   
    
    lambda_0 = 0
    
    expression = lambda_0 * w   
    
    part1 = 1-np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))])
    #print(1+math.exp(-y[i]*(x[i] @ w)))
    
    part2 = y * part1
    
    #print(np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))]),part2)
    
    expression = expression - x.T @ part2

    print(expression)       
            
    return expression

def double_prime(w,x,y):
    
    lambda_0 = 0
    
    expression = lambda_0*np.identity(3)
    
    part1 = np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))])
    
    part2 = part1 * (1-part1)
    
    part3 = np.diag(part2)
    
    expression = expression + x.T @ (part3 @ x)
    #print(np.sum(expression,axis = 0))     
            
    return expression

def Newton(x0,x,y):
    
    iteration = 0
    
    while iteration <1000000:
        
        hess_inv = np.linalg.inv(double_prime(x0,x,y))
        
        x1 = x0 - hess_inv @ gradient(x0,x,y)
        
        
        if np.linalg.norm(gradient(x0,x,y)) > 0.0000001:
            
            x0=x1            
        #print(x1,gradient(x0,x,y))
        
            iteration = iteration + 1
            
        else:
            break
       
        #x0 = x1
    
    return x0, iteration

if __name__ == "__main__": 

    x0, iteration = Newton(x0,x,y)    
        
    