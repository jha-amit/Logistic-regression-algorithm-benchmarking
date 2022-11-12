# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:54:00 2022

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

x0 = np.array([0,0,0])



    
x = np.array([[1,3,0],[2,4,0],[2.5,3.5,0],[3,1.5,0],[3.5,4.5,0],[4,1.5,0],\
              [4,3.5,0],[4.5,2.5,0],[5,1,0],[6,2,0],[6.5,3,0],[7,1.5,0]])
    
y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1,1,-1])


def function(w,x,y):
    
    lambda_0 = 1
    
    part1 = np.zeros(len(y))
    
    exponent = np.array([-y[i]*(x[i] @ w) for i in range(len(y))])
    
    expression = lambda_0*1/2 * np.dot(w,w)
    
    for i in range(len(y)):
        
        if exponent[i] > 20:
            
            part1[i] = 0
            
        elif exponent[i] < -20:
            
            part1[i] = 10000000            
        
        else:       
            part1[i] = math.log(1+math.exp(-y[i]*(x[i] @ w)))
            
        expression = expression + part1[i]
    
    return expression

def gradient(w,x,y):   
    
    lambda_0 = 1
    
    expression = lambda_0 * w
    
    exponent = np.array([-y[i]*(x[i] @ w) for i in range(len(y))])
    
    part1 = [1]*len(y)
    
    for i in range(len(y)):
        
        if exponent[i] > 20:
            
            part1[i] = 1
            
        elif exponent[i] < -20:
            
            part1[i] = 0
        else:           
    
            part1[i] = 1 - 1/(1+math.exp(-y[i]*(x[i] @ w)))
    
    
    part2 = y * np.array(part1)
    
    #print(np.array(part1))
    
    #print(np.array([-y[i]*(x[i] @ w) for i in range(len(y))]),part2)
    
    expression = expression - x.T @ part2

    #print(expression)       
            
    return expression

def double_prime(w,x,y):
    
    lambda_0 = 1
    
    expression = lambda_0*np.identity(3)
    
    part1 = np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))])
    
    part2 = part1 * (1-part1)
    
    part3 = np.diag(part2)
    
    expression = expression + x.T @ (part3 @ x)
    #print(np.sum(expression,axis = 0))     
            
    return expression


def DFP_method(x0,x,y):
       
    iteration = 0
    gamma = 0.5
  
    H0 = np.identity(len(x0))
    epsilon = 1e-6
    
    g0 = gradient(x0,x,y)   
    
    sigma = 0.5
    H = H0
    while np.linalg.norm(g0) > epsilon:  
    
        d0 = -H @ g0
        
        lambda_0 = 1.1
        
        x1 = x0 + lambda_0 * d0       
        
        p_k = x0 - x1
        
        g1 = gradient(x1,x,y)      
        
        q_k = g0 - g1
        
        
        if gamma * np.linalg.norm(p_k)*np.linalg.norm(q_k) <= np.dot(p_k,q_k):
            
            H = H + (np.matrix(p_k).T @ np.matrix(p_k)) / (p_k @ q_k) - (np.matrix(H) @ np.matrix(q_k).T)\
                @ (np.matrix(q_k) @ np.matrix(H))/ (q_k @ (np.matrix(H) @ np.matrix(q_k).T))
                
    
            H = np.array(H)
            
        else:
            H = H0        
        g0 = g1
        
        x0= x1
        
        iteration = iteration + 1
        print(iteration)
    return x0, iteration

        
########################################################################

start_time = time.time()
# DFP
if __name__ == "__main__": 
    x0, iteration = DFP_method(x0,x,y)
    
end_time = time.time()
print(end_time - start_time)



