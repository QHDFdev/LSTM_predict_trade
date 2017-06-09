#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:46:44 2017

@author: wanjun
打标签
t分钟内的收盘价比此时刻的收盘价
总共有三种标签

"""
#%%
#导入相关库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import talib 
#%%
#自定义函数
#提取训练集指标最大值
def train_index_max_min(data):
    lst_max=pd.Series(index=data.columns)
    lst_min=pd.Series(index=data.columns)
    for i in data.columns:
        lst_max[i]=max(data[i])
        lst_min[i]=min(data[i])
    return lst_max,lst_min
#打标签
def p(x):
    dot=1
    for i in x:
        dot*=i
    return dot

def f(x):
    if x>c:
        return 1
    else:
        return 0
def g(x):
    if x>c:
        return -1
    else:
        return 0
def h(x):
    count=0
    for i in x:
        if i==1:
            count+=1
    return count
def mark_label(data,t):
    label=pd.DataFrame(index=data.index,columns=range(t+1)[1:])
    label_final=label.copy()
    temp=data['close'].copy()   
    for i in range(t+1)[1:]:
        label[i]=abs(temp.shift(-i)-temp)-(temp.shift(-i)*b+temp*a)
        label_final[i][(temp.shift(-i)-temp)>0]=label[i][(temp.shift(-i)-temp)>0].apply(f)
        label_final[i][(temp.shift(-i)-temp)<0]=label[i][(temp.shift(-i)-temp)<0].apply(g)
        label_final[i][(temp.shift(-i)-temp)==0]=0
    return label_final

def count_label(data):
    label=pd.DataFrame(index=data.index,columns=['label'])
    label[((data==1).apply(h,axis=1)>=5)]=1
    label[((data==-1).apply(h,axis=1)>=5)]=-1
    return label
#%%
# 导入原始数据数据
file='/Users/wanjun/Desktop/LSTM模型/data/data_2017.csv'
data=pd.read_csv(file,index_col='datetime',parse_dates=True)
#%%
#参数设置
global a,b
c=0    #打标签的阀值
a=0.00024   #开仓的手续费
b=0.00092   #平仓的手续费用
t=30        #未来t分钟内都打标签
size=len(data)   #选择数据集的大小
#%%
#计算标签
label=mark_label(data,t)
label=label.dropna()
#选择第t分钟的打标签
for i in range(t+1)[1:]:
    if i!=t:
        del label[i]
label.columns=['label']
#%%
#增加指标
arrOpen=np.array(data.open)
arrClose = np.array(data.close)
arrHigh = np.array(data.high)
arrLow = np.array(data.low)
arrVolume = np.array(data.volume,dtype=np.float)

#EMA,RSI指标
data['EMA_5min'] = talib.EMA(arrClose, timeperiod=5)
data['EMA_10min'] = talib.EMA(arrClose, timeperiod=10)
data['EMA_15min'] = talib.EMA(arrClose, timeperiod=15)
data['EMA_20min'] = talib.EMA(arrClose, timeperiod=20)
data['RSI'] = talib.RSI(arrClose)
#data['STOCHRSI_usual'] = talib.STOCHRSI(arrClose)

# BOLL-BAND指标
BBANDS_usual = talib.BBANDS(arrClose)
upperband, middleband, lowerband = BBANDS_usual
data['upperband'] = upperband
data['middleband'] = middleband
data['lowerband'] = lowerband


# MACD指标
MACD_usual = talib.MACD(arrClose)
macd, macdsignal, macdhist = MACD_usual
data['macd'] = macd
data['macdsignal'] = macdsignal
data['macdhist'] = macdhist

# KDJ指标
KDJ_usual = talib.STOCH(arrHigh, arrLow, arrClose)
slowk, slowd = KDJ_usual
data['slowk'] = slowk
data['slowd'] = slowd

# ATR指标
ATR_usual   = talib.ATR(arrHigh, arrLow, arrClose)
data['ATR'] = ATR_usual

# WILLR指标
WILLR_usual = talib.WILLR(arrHigh, arrLow, arrClose)
data['WILLR'] = WILLR_usual

# BOV指标
OBV_usual  = talib.OBV(arrClose, arrVolume)
data['OBV'] = OBV_usual

# SAR指标
SAR_usual  = talib.SAR(arrHigh, arrLow)
data['SAR'] = SAR_usual

# DEMA指标
DEMA_usual = talib.DEMA(arrClose)
data['DEMA'] = DEMA_usual

#MOM指标
MOM_usual  = talib.MOM(arrClose)
data['MOM'] = MOM_usual
"""  
#DEMA
DEMA=talib.DEMA(arrClose)
data['DEMA_30min']=DEMA
    
#WMA
WMA=talib.WMA(arrClose)
data['WMA_30min']=WMA
    
#ADX
ADX=talib.ADX(arrHigh,arrLow,arrClose,timeperiod=30)
data['ADA_30min']=ADX
    
#APO
APO=talib.APO(arrClose, fastperiod=12, slowperiod=26, matype=0)
data['APO']=APO
    
#AROON
aroondown, aroonup =talib.AROON(arrHigh, arrLow, timeperiod=30)
data['aroondown']=aroondown
data['aroonup']=aroonup

#BOP
BOP=talib.BOP(arrOpen,arrHigh,arrLow,arrClose)
data['BOP']=BOP

#CCI
CCI=talib.CCI(arrHigh,arrLow,arrClose,timeperiod=30)
data['CCI_30']=CCI
    
#DX
DX=talib.DX(arrHigh,arrLow,arrClose,timeperiod=30)
data['DX_30']=DX
    
#MFI
MFI=talib.MFI(arrHigh,arrLow,arrClose,arrVolume,timeperiod=30)
data['MFI_30']=MFI

#MINUS_DI
MINUS_DI = talib.MINUS_DI(arrHigh, arrLow, arrClose, timeperiod=30)
data['MINUS_DI_30']=MINUS_DI

#MINUS_DM 
MINUS_DM  = talib.MINUS_DM(arrHigh, arrLow, timeperiod=30)
data['MINUS_DM ']=MINUS_DM 
"""
#%% 
#数据处理和归一化,提取每个指标的最大值和最小值
#data['open']=data.open.diff()
#data['high']=data.high.diff()
#data['low']=data.low.diff()
#data['close']=data.close.diff()
data=data.dropna()
data=data[:-t]
m=len(label)-len(data)
label=label[m:]
#%%存取数据
lst_max,lst_min=train_index_max_min(data)
lst_max=pd.DataFrame(lst_max,index=lst_max.index,columns=['number'])
lst_min=pd.DataFrame(lst_min,index=lst_min.index,columns=['number'])
data=data[-size:]
label=label[-size:]
data=data.apply(lambda x:preprocessing.minmax_scale(x))
data.to_csv('data.csv')
label.to_csv('label.csv')
lst_max.to_csv('lst_max.csv')
lst_min.to_csv('lst_min.csv')
