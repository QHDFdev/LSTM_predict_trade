#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:15:26 2017

@author: wanjun
时间序列聚类
"""

import pandas as pd
import numpy as np
import  sklearn.cluster
from sklearn import preprocessing
from matplotlib import pyplot as plt

file='/Users/wanjun/Desktop/LSTM模型/bar_2016_zhuli_IF.csv'
data=pd.read_csv(file,index_col=1,parse_dates=True)
data.drop('Unnamed: 0',inplace=True,axis=1)
close=data.close.copy()

#%%
#对数据进行正则化
T=30
t=1
n=len(close)
i=0
new_data=pd.DataFrame(columns=range(T))
new_close=pd.DataFrame(columns=range(T))
while i+T<n:
    new_close=new_close.append(pd.DataFrame(close.iloc[i:i+T].values.reshape(1,30),columns=range(T),index=[close.index[i]]))
    temp=pd.DataFrame(preprocessing.normalize(close.iloc[i:i+T].values.reshape(1,30)),columns=range(T),index=[close.index[i]])
    i+=t
    new_data=new_data.append(temp)
#%%
clf=sklearn.cluster.KMeans(5)
label=clf.fit_predict(new_data)
#%%
lst2=np.where(label==2)[0]
for i in lst2:
    new_data.iloc[i].plot()
#%%
new_label=pd.DataFrame(label,index=new_close.index,columns=['label'])
for i in range(len(new_label)):
    if new_label.iloc[i].label==0 or new_label.iloc[i].label==2:
        new_label.iloc[i].label=-1
    elif new_label.iloc[i].label==1 or new_label.iloc[i].label==3:
        new_label.iloc[i].label=1
    else:
        new_label.iloc[i].label=0
    
    