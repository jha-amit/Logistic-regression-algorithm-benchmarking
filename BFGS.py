# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:10:04 2022

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


x = np.array([[1,0,3],[2,1,4],[2.5,.5,3.5],[3,1.5,1.5],[3.5,2,4.5],[4,2,1.5],\
              [4,2,3.5],[4.5,1.5,2.5],[5,3,1],[6,1,2],[6.5,3,3],[7,3.5,1.5]]) 
    
x = np.array([[1,3,1],[2,4,1],[2.5,3.5,1],[3,1.5,1],[3.5,4.5,1],[4,1.5,-1],\
              [4,3.5,-1],[4.5,2.5,-1],[5,1,-1],[6,2,-1],[6.5,3,1],[7,1.5,-1]])
    
x = np.array([[1,3,0],[2,4,0],[2.5,3.5,0],[3,1.5,0],[3.5,4.5,0],[4,1.5,0],\
              [4,3.5,0],[4.5,2.5,0],[5,1,0],[6,2,0],[6.5,3,0],[7,1.5,0]])
    
y = np.array([1,1,1,1,1,-1,-1,-1,-1,-1,1,-1])


def function(w,x,y):
    
    lambda_0 = 2
    
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
    
    lambda_0 = 2
    
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
    
    lambda_0 = 0
    
    expression = lambda_0*np.identity(3)
    
    part1 = np.array([1/(1+math.exp(-y[i]*(x[i] @ w))) for i in range(len(y))])
    
    part2 = part1 * (1-part1)
    
    part3 = np.diag(part2)
    
    expression = expression + x.T @ (part3 @ x)
    #print(np.sum(expression,axis = 0))     
            
    return expression


def BFGS_method(x0,x,y):
    rho = 1/2    
    iteration = 0
    gamma = 0.5
    alpha =.1
    beta = 0.9
    B0 = np.identity(len(x0))
    epsilon = 1e-6
    g0 = gradient(x0,x,y)
    sigma = 0.5
    B = B0
    ita =.01
    
    while np.linalg.norm(g0) > epsilon:    
       
        d0 = -np.linalg.inv(B) @ g0       
        
        #lambda_0 = ARMIJO_search(x0,x,y,rho,alpha,beta,sigma)
        
        lambda_0 = 1 #steepest_decent(x0,ita,x,y)
        
        #direction = -g0/np.linalg.norm(g0)
        
        # if np.linalg.norm(gradient(lambda_0,x,y)) < epsilon:
        #     break
    
        x1 = x0 + lambda_0 * d0 
        
        #print(lambda_0)
        
        p_k = x0 - x1
        
        g1 = gradient(x1,x,y)
        
        
        #print(g1,x1)            
       
        q_k = g0 - g1
        
        #print(d0)
        
        if gamma * np.linalg.norm(p_k)*np.linalg.norm(q_k) <= np.dot(p_k,q_k):
            
            B = B + (np.matrix(q_k).T @ np.matrix(q_k)) / (p_k @ q_k) - (np.matrix(B) @ np.matrix(p_k).T)\
                @ (np.matrix(p_k) @ np.matrix(B))/ (p_k @ (np.matrix(B) @ np.matrix(p_k).T))
                
            B = np.array(B)
            
         
            #print(np.linalg.norm(g1),B)
        else:
            B = B0        
        g0 = g1
        
        x0= x1
        
        iteration = iteration + 1
        print(iteration)
    return x0, iteration

        
########################################################################

# line search

def steepest_decent(x0,ita,x,y):
    iteration = 0 
    
    while iteration <1000000:
        
        
        print(iteration,gradient(x0,x,y),x0)
        
        if np.linalg.norm(gradient(x0,x,y)) > 0.00001:
            
            dk = -ita * gradient(x0,x,y)
            x0 = x0 + dk
            iteration = iteration + 1
            #print(iteration)
            
        else:        
            break
        
    return x0
   

def ARMIJO_search(x0,x,y, rho, alpha, beta,sigma):
    
    g0 = gradient(x0,x,y)
    
    direction = -g0/np.linalg.norm(g0)
    
    lambda_0 = 0.1 #sigma * np.linalg.norm(g0) * np.linalg.norm(direction)
    
    lambda_k = [lambda_0*(rho**k) for k in range(0,10)]    
    
    phi_0 = function(x0,x,y) # as per the function
    
    phi_prime_0 = g0 @ direction
    
    #phi_prime_0 = -0.5 * np.linalg.norm(gradient(x0,x,y))             
            
    fun = [function((x0 + lambda_k[i] * direction),x,y) for i in range(len(lambda_k))]

    phi_prime = [gradient((x0 + lambda_k[i] * direction),x,y) @ direction for i in range(len(lambda_k))]   
    
    lambda_opt = []
    
    #print(phi_prime,phi_prime_0,direction)
    
    for fun_val, lambda_val, phi_pr  in zip(fun,lambda_k,phi_prime):
        
        #print(phi_pr)
        
        if (fun_val <= phi_0 + alpha*lambda_val*phi_prime_0) and (phi_pr >= beta * phi_prime_0):
            
            lambda_opt.append(lambda_val)
            
    if lambda_opt:
        
        lambda_optimum = max(lambda_opt)
    
    else : 
       
        lambda_optimum = 'value not found change alpha'
        
 
        
    return lambda_optimum

# BFGS
start_time = time.time() 
x0, iteration = BFGS_method(x0,x,y)
end_time = time.time()
print(end_time - start_time)




