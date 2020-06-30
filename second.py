# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:28:58 2020

@author: User
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
def gradient(f,q,yy,ytest):
    step=10
    a=0.001
    for i in range(0,1):
        yp=f @ q.T
        co=1/2*o*np.power(np.sum(yp-yy),2)
        q = q - (a/o) * np.sum( (yp - yy)*f,axis=0)   
    ax2 = sns.distplot(ytest, hist=False, color="r", label="Actual Value")
    sns.distplot(xtest @ q.T, hist=False, color="b", label="Fitted Values" , ax=ax2)
    sxp=xtest @ q.T
    sxp=sxp.reshape(10,)
    sx1=xtest[:,1]
    sx2=xtest[:,2]
    ytest=ytest.reshape(10,)
    ax.scatter(sx1,sx2,ytest,zdir='z')
    ax.plot(sx1,sx2,sxp)
        

x=pd.read_csv('ex1data2.txt',names=["size","bedrooms","price"])
m=x['size'].size
features = (np.stack((np.ones(m),(x['size']-np.mean(x['size']))/np.std((x['size'])),(x['bedrooms']-np.mean(x['bedrooms']))/np.std(x['bedrooms'])),axis=1))
q= np.ones([1,3])
y = x.iloc[:,2:3].values 
y=(y-y.mean())/y.std()
xtrain,xtest,ytrain,ytest=train_test_split(features,y,test_size=0.20,random_state=0)
o=ytrain.size
l1=np.array((x['size']-x['size'].mean())/x['size'].std())
l2=np.array((x['bedrooms']-x['bedrooms'].mean())/x['bedrooms'].std())
l3=np.array((x['price']-x['price'].mean())/x['price'].std())

gradient(xtrain,q,ytrain,ytest) 