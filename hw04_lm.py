# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:20:11 2016

@author: Cicely
"""

import numpy as np
import myLM
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

#函数形式为a*sin(2*pi*b*x + c)
def my1Func_sin(x,p):
    return p[0]*np.sin(2*np.pi*p[1]*x + p[2])
    
def gradient_sin(x,p):
    g = np.zeros([1, len(p)])       
    g[0][0] = np.sin(2*np.pi*p[1]*x + p[2])
    g[0][1] = p[0]*np.cos(2*np.pi*p[1]*x + p[2])*2*np.pi*x
    g[0][2] = p[0]*np.cos(2*np.pi*p[1]*x + p[2])
   
    return g

def residuals_sin(p,y,x):
    return y - my1Func_sin(x,p)



#函数形式为 a*e^-bx
def my1Func_exp(x,p):
    return p[0]*np.exp(-p[1]*x)
    

def gradient_exp(x,p):
    g = np.zeros([1, len(p)])       
    g[0][0] = np.exp(-p[1] * x)
    g[0][1] =  -p[0] * x * np.exp(-1* p[1] * x)
   
    return g
    
    
def residuals_exp(p,y,x):
    return y - my1Func_exp(x,p)



if __name__=="__main__":
    
    obs_x = np.linspace(0,-2*np.pi,30)
    p = [5,0.34,np.pi/3]
    y0 = myLM.myFunc(obs_x,p,my1Func_sin) 
    obs_y = y0 + 2*np.random.randn(len(obs_x))
    
    p0=[1,-2,1]
    M_K = myLM.MyLM(my1Func_sin,gradient_sin,obs_x,obs_y,p0,2,100)
    p_lsq = leastsq(residuals_sin,p0,args=(obs_y,obs_x))
    print "leastsq method:",p_lsq[0]
    x_show = np.linspace(0,-2*np.pi,100)
    plt.plot(x_show, myLM.myFunc(x_show,p,my1Func_sin) ,label='real function')
    plt.plot(obs_x,obs_y,'bo',label='data with noise')
    plt.plot(x_show,myLM.myFunc(x_show,M_K,my1Func_sin),label='fitted function',linewidth=4)
    plt.plot(x_show,myLM.myFunc(x_show,p_lsq[0],my1Func_sin),label='fitted function leastsq')
    plt.legend()
    plt.show()
    
    
    """
    """
    obs_x = np.linspace(1,8,10)
    p = [5,0.5]
    y0 = myLM.myFunc(obs_x,p,my1Func_exp) 
    obs_y = y0 + 2*np.random.randn(len(obs_x))
    
    p0=[10,2]
    M_K = myLM.MyLM(my1Func_exp,gradient_exp,obs_x,obs_y,p0,2,100)
    p_lsq = leastsq(residuals_exp,p0,args=(obs_y,obs_x))
    print "leastsq method:",p_lsq[0]
    x_show = np.linspace(1,8,300)
    plt.plot(x_show, myLM.myFunc(x_show,p,my1Func_exp) ,label='real function')
    plt.plot(obs_x,obs_y,'bo',label='data with noise')
    plt.plot(x_show,myLM.myFunc(x_show,M_K,my1Func_exp),label='fitted function',linewidth=3)
    plt.plot(x_show,myLM.myFunc(x_show,p_lsq[0],my1Func_exp),label='fitted function leastsq')
    plt.legend()
    plt.show()


    