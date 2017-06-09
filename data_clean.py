#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:35:05 2017

@author: wanjun
数据清洗
"""
#%%
import numpy as pd
import pandas as pd
from sklearn import preprocessing
import talib 
from  matplotlib import pyplot as plt
#%%
file='/Users/wanjun/Desktop/LSTM模型/data/data_train_latest.csv'
file_1='/Users/wanjun/Desktop/LSTM模型/data/MINUTE_zhuli_IF_20170509.csv'
data=pd.read_csv(file)
data_1=pd.read_csv(file_1,index_col=1,parse_dates=True)
data.index=data_1.index
data['datetime']=data_1.index
#%%
data=data.sort_index()
data=data['2016-12-19':]
index=data.drop_duplicates('datetime').resample('D').mean().dropna().index
data_clean=pd.DataFrame(columns=['open','high','low','volume','close'])
lst_len=[]
lst=[]
for i in index:
    i=str(i)[:10]
    temp=data[i]
    start=i+' 09:30:00'
    end=i+' 15:01:00'
    data_clean=data_clean.append(temp[start:end])
    if len(temp[start:end])>242 or len(temp[start:end])<241:
        lst.append(i)
    lst_len.append(len(temp[start:end]))
    
#%%
#11月18日单独作处理，因为超过242,经过分析删除这一行
#data_clean=data_clean.drop('2016-11-18 13:01:00')
#%%
data_clean_1=pd.DataFrame(columns=['open','high','low','volume','close'])
#%%
for i in index:
    start1=str(i)+' 09:30:00'
    end1=str(i)+' 11:30:00'
    start2=str(i)+' 13:00:00'
    end2=str(i)+' 15:00:00'
    index_new=pd.date_range(start1,end1,freq='min').append(pd.date_range(start2,end2,freq='min'))
    i=str(i)[:10]
    temp=data_clean[i].copy()
    temp.index=index_new[:len(temp)]
    data_clean_1=data_clean_1.append(temp)

del data_clean_1['datetime'] 
      
data_clean_1.to
    
    
    
    
    
    
    