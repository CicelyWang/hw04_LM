# -*- coding: utf-8 -*-
"""
Created on Thu May 05 19:08:57 2016

@author: Cicely
"""

from ad import *
import numpy as np

e1 = 0.001
e2 = 0.0001

def myNorm(v):
    sum = 0
    for i in v:
        sum += i ** 2
    return np.sqrt(sum)
    

def myFunc(x,p,func):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = func(x[i],p)
    return y


    
def Jacobian(x,p,g_func):
    J = np.zeros([len(x), len(p)])
    for j in range(len(x)):
        J[j]= g_func(x[j],p)
    return J
    
def MyLM(func,g_func, obs_x, obs_y, p0,  v, maxIter):
   
    p = p0
    func_y = myFunc(obs_x,p,func)
    d = obs_y - func_y
    J = Jacobian(obs_x,p,g_func)
    Hs = np.dot(J.T,J)
    g = np.dot(J.T, d)
    if myNorm(g) <= e1:
         print p, 
         print myNorm(d)
         return p
    u = Hs[0][0]
    
    for i in range(len(p0)):
        if Hs[i][i] > u:
            u = Hs[i][i]
   
    iter = 0       
    while  iter < maxIter:
        iter += 1
        print iter,":",
        
        while 1:
            sp = np.dot(np.linalg.inv(Hs + u * np.eye(len(p))),g)
            if myNorm(sp) <= e2* myNorm(p):
                print p, 
                print myNorm(d)
                return p
                
            else:           
                r =  ( np.square(myNorm(d)) - np.square(myNorm(obs_y - myFunc(obs_x,p+sp,func))) )/ np.dot(sp.T, (u * sp + g))
                if r > 0 :
                    p = p + sp
                    ratio = 1 - (2 * r - 1)**3
                    if ratio < 1.0 / 3:
                        ratio = 1.0/3
                    u = u * ratio
                    v = 2 
                    func_y = myFunc(obs_x,p,func)
                    d = obs_y - func_y
       
                    J = Jacobian(obs_x,p,g_func)
                    Hs = np.dot(J.T,J)
                    g = np.dot(J.T, d)
                    if myNorm(g) <= e1:
                        print p,
                        print myNorm(d)
                        return p
                    break
                else :
                    u = u * v
                    v = 2 * v
            #sp的大小
        print p,
        print myNorm(d)
   
    return p




