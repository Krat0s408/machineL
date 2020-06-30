# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:20:15 2020

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient(s,c,n):
  theta0 = 0 
  theta1 = 0
  step=int(1500)
  a=0.01
  yp=0
  z=[]
  k=[]
  
  
  for i in range(0,step):
    yp = theta0+theta1*s
    co = 1/(2*n)*sum((yp-c)**2)
    z.append(co)
    d1=(1/n)*sum((yp-c))
    d2=(1/n)*sum((yp-c)*s)
    temp0=theta0-a*d1
    temp1=theta1-a*d2
    theta0=temp0
    theta1=temp1
    k.append(i)
    
    
  plt.scatter(x['population'],x['profit'],color='orange')
  plt.plot(s,yp,color='red')
  plt.xlabel('population')
  plt.ylabel('profit')
  

x = pd.read_csv('Desktop\machine-learning-ex1\ex1\ex1data1.txt',header=None)
x.columns = ['population','profit']
si = np.array(x['population'])
pr = np.array(x['profit'])
n=si.size
gradient(si,pr,n)