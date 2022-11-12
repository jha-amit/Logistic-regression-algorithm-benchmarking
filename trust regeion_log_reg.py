# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:49:29 2022

@author: amit
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:02:21 2022

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



    
x = np.array([[1,3,1],[2,4,0],[2.5,3.5,1],[3,1.5,1],[3.5,4.5,-1],[4,1.5,1],\
              [4,3.5,0],[4.5,2.5,0],[5,1,0],[6,2,0],[6.5,3,0],[7,1.5,0]])
    
y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1,1,-1])


def function(w,x,y):    
    lambda_0 = 1
    expression = lambda_0 * (1/2) * np.dot(w,w)
    for i in range(len(y)):
        expression =  expression + math.log(1+math.exp(-y[i]*np.dot(w,x[i])))
                
    return expression

def gradient(w,x,y):   
    
    lambda_0 = 1
    
    expression = lambda_0 * w   
    
    part1 = 1-np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))])
    #print(1+math.exp(-y[i]*(x[i] @ w)))
    
    part2 = y * part1
    
    #print(np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))]),part2)
    
    expression = expression - x.T @ part2

    print(expression)       
            
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


def trust_regeion(x0,x,y):
    
    lambda_0 = 0 
    delta_0 = 0.25
    iteration = 0
    epsilon = 0.0000001
    
    while iteration<100:
        
        lambda_0 = 0    
        
        d =np.array([0.5/math.sqrt(2),0.5/math.sqrt(2)])
        
        delta_f = gradient(x0,x,y)
        
        delta_2_f = double_prime(x0,x,y)
        
        print(delta_2_f)
        
        A = delta_2_f + lambda_0
        
        print(iteration, A)
        
        L = scipy.linalg.cholesky(A, lower=True)
        
        d_lambda = -np.dot(np.linalg.inv(A),delta_f)
        
        mod_d_lambda = np.linalg.norm(d_lambda)
        
        w = np.dot(np.linalg.inv(L),d_lambda)
        
        mod_w = np.linalg.norm(w)
        
        lambda_1 = lambda_0 + (1-mod_d_lambda/delta_0)*(-(mod_d_lambda)**2/mod_w)
        
        A = delta_2_f + lambda_1
        
        L = scipy.linalg.cholesky(A, lower=True)
        
        d_lambda1 = -np.dot(np.linalg.inv(A),delta_f)
        
        mod_d_lambda1 = np.linalg.norm(d_lambda1)
        
        fun_x0 = function(x0,x,y)
        
        phi_xd = fun_x0 + np.dot(delta_f,d_lambda1) +\
            np.dot(d_lambda1,np.dot(delta_2_f,d_lambda1))
        
        x1 = x0 + d_lambda1
        
        fun_x1 = function(x1,x,y)
        r = (fun_x0 - fun_x1)/(fun_x0 - phi_xd)
        
      
        if r<0.25:
            print(r,iteration, x0)
            delta_0 = delta_0/2   
            
        else:
            if r>0.75 and  mod_d_lambda1== delta_0:
                delta_0 = 2*delta_0       
            
            else:
                delta_0 = delta_0
        
        if np.linalg.norm(gradient(x0,x,y)) < epsilon:
            break
        
        if r > 0.001:
            x0 = x1
            
        else:
            x0 = x0
            
        iteration = iteration + 1
        
    return x0,iteration

x_opt,iteration = trust_regeion(x0,x,y)
        
        
      
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        