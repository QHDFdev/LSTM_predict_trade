#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:45:23 2017

@author: wanjun

时间序列分析
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF


file='/Users/wanjun/Desktop/LSTM模型/bar_2016_zhuli_IF.csv'
data=pd.read_csv(file,index_col=1,parse_dates=True)
data.drop('Unnamed: 0',inplace=True,axis=1)

#查看特征间的相关系数
data.corr()
#自相关图
fig=sm.graphics.tsa.plot_acf(data.close[:5000])
#偏自相关图
fig=sm.graphics.tsa.plot_pacf(data.close[:5000])
