#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 18:10:45 2017

@author: wanjun
对label的数据分析
"""
#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
#自定义函数
#%%
def f(x):
    count=0
    for i in x:
        if i==1:
            count+=1
    return count
#%%
#数据读入  
file_data='/Users/wanjun/Desktop/LSTM模型/code/label_raw.csv' 
data=pd.read_csv(file_data,index_col='datetime',parse_dates=True)
label=pd.DataFrame(index=data.index,columns=['label'])
#数据分析处理
label[(data==-1).apply(f,axis=1)>=7]=-1
label[(data==1).apply(f,axis=1)>=7]=1
label=label.fillna(0)
label.to_csv('label.csv')
#%%
file_data='/Users/wanjun/Desktop/LSTM模型/data/data_IF.csv'
data=pd.read_csv(file_data,index_col='Time',parse_dates=True)
a=data.Volume.copy()
def f(x):
    if x>2000:
        return 2000
    elif x<200:
        return 200
a=a.apply(f)
        