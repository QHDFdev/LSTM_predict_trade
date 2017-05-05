
# -*- coding: utf-8 -*-
"""

@author: wanjun
LSTM平台测试
"""

#coding: UTF-8
#%%
#导入相关库
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
import tarfile
import talib
from sklearn import preprocessing
import time
import datetime
from collections import defaultdict
#%%
def on_init(context):   
    #获取当前时刻的时间和交易日期
    datetime,date = context.function.get_time()
    #解压模型
    context.function.log('init')
    tar=tarfile.open('LSTM.tar.gz')
    tar.extractall()
    tar.close()
    context.function.log('解压成功')

    #模型的参数 （如果参数改变，则训练的模型也要改变）
    #平台参数
    global T
    T=100 #每天前T分钟收集数据
    #网络参数
    global n_input,n_steps,n_hidden,n_classes
    n_input = 24 # 时间序列点的特征数
    n_steps = 60 # 时间序列的长度
    n_hidden = 120 # 隐藏层结点数
    n_classes = 3 # 标签数量
    #训练集合的各种指标的最大值和最小值（以此作为归一化的上限和下限）
    global lst_max,lst_min
    path='lst_max.csv'
    lst_max=pd.read_csv(path)
    path='lst_min.csv'
    lst_min=pd.read_csv(path)
    context.function.log('装载训练集最大值成功')
    #每分钟的基础指标 
    context.var.ts=pd.DataFrame(columns=['open','close','high','low','volume','EMA_5min',
       'EMA_10min', 'EMA_15min', 'EMA_20min', 'RSI', 'upperband',
       'middleband', 'lowerband', 'macd', 'macdsignal', 'macdhist',
       'slowk', 'slowd', 'ATR', 'WILLR', 'OBV', 'SAR', 'DEMA', 'MOM'])
    context.var.pred=[]
    context.var.real=[]
    #计数指标
    context.var.count=0
    #创建tensorflow图框架
    global x,y,weights,biases,pred
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # 定义W,B 
    weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    pred = RNN(x, weights, biases)
    #装载模型
    global saver
    saver = tf.train.Saver()
    context.function.log('装载模型成功')
#%%
#自定义函数
#生成lstm    
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.8)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # 输出函数使用的是线性函数
    # 时间序列的最后一个作为输出
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
#计算指标
def technical_index(context):
    context.var.ts
    #EMA,RSI指标
    context.var.ts['EMA_5min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=5)
    context.var.ts['EMA_10min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=10)
    context.var.ts['EMA_15min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=15)
    context.var.ts['EMA_20min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=20)
    context.var.ts['RSI'] = talib.RSI(np.array(context.var.ts.close))
    STOCHRSI_usual = talib.STOCHRSI(np.array(context.var.ts.close))
    
    # BOLL-BAND指标
    BBANDS_usual = talib.BBANDS(np.array(STOCHRSI_usual.close))
    upperband, middleband, lowerband = BBANDS_usual
    context.var.ts['upperband'] = upperband
    context.var.ts['middleband'] = middleband
    context.var.ts['lowerband'] = lowerband
    
    arrClose = np.array(context.var.ts.close)
    arrHigh = np.array(context.var.ts.high)
    arrLow = np.array(context.var.ts.low)
    arrVolume = np.array(context.var.ts.volume,dtype=np.float)
    # MACD指标
    MACD_usual = talib.MACD(arrClose)
    macd, macdsignal, macdhist = MACD_usual
    context.var.ts['macd'] = macd
    context.var.ts['macdsignal'] = macdsignal
    context.var.ts['macdhist'] = macdhist
    
    # KDJ指标
    KDJ_usual = talib.STOCH(arrHigh, arrLow, arrClose)
    slowk, slowd = KDJ_usual
    context.var.ts['slowk'] = slowk
    context.var.ts['slowd'] = slowd
    
    # ATR指标
    ATR_usual   = talib.ATR(arrHigh, arrLow, arrClose)
    context.var.ts['ATR'] = ATR_usual
    
    # WILLR指标
    WILLR_usual = talib.WILLR(arrHigh, arrLow, arrClose)
    context.var.ts['WILLR'] = WILLR_usual
    
    # BOV指标
    OBV_usual  = talib.OBV(arrClose, arrVolume)
    context.var.ts['OBV'] = OBV_usual
    
    # SAR指标
    SAR_usual  = talib.SAR(arrHigh, arrLow)
    context.var.ts['SAR'] = SAR_usual
    
    # DEMA指标
    DEMA_usual = talib.DEMA(arrClose)
    context.var.ts['DEMA'] = DEMA_usual
    
    #MOM指标
    MOM_usual  = talib.MOM(arrClose)
    context.var.ts['MOM'] = MOM_usual
#数据标准化：归一化
def standard(temp):
    def f(x,max,min):
        for j,i in enumerate(x):
            if i>=max:
                x[j]=1
            elif i<=min:
                x[j]=0
            else:
                x[j]=float((i-min)/(max-min))
            return x
    for char in temp.columns:
        temp[char]=temp[char].apply(lambda x: f(x,lst_max.ix[char],lst_min.ix[char]))
#提取时间序列函数
def get_ts(temp):
    lst=[]
    return lst.append(temp[-n_steps:].values.tolist())
#转换预测值为标签
def convert(predict):
    n=np.argmax(predict)
    if n==0:
        return "可买涨"
    elif n==1:
        return "观望"
    else:
        return "可买跌"        
#%%       
def on_start(context):
    context.function.log(u'LSTM start')
    context.var.beginTrace = True
    #清理每天的数据
    context.var.ts=pd.DataFrame(columns=['open','close','high','low','volume','EMA_5min',
       'EMA_10min', 'EMA_15min', 'EMA_20min', 'RSI', 'upperband',
       'middleband', 'lowerband', 'macd', 'macdsignal', 'macdhist',
       'slowk', 'slowd', 'ATR', 'WILLR', 'OBV', 'SAR', 'DEMA', 'MOM'])
    context.var.pred=[]
    context.var.real=[]
    context.var.count=0
    context.function.log(u'clear data')
def on_stop(context):
    context.function.log(u'LSTM stoped')
    context.var.beginTrace = False
def on_tick(context):
    pass
#%%  
def on_bar(context):
    if not context.var.beginTrace:
        return
    context.var.count+=1
    context.function.log(context.var.count)
    #获取当前时刻的bar数据
    bar=context.var.bar
    #组合得到新的ts
    df=pd.DataFrame(np.random.randn(1,24),columns=context.var.ts.columns)
    df['close']=bar.close
    df['open']=bar.open
    df['low']=bar.low
    df['high']=bar.high
    df['volume']=bar.volume
    context.var.ts=context.var.ts.append(df,ignore_index=True)
    technical_index(context)
    temp=context.var.ts
    temp['close']=temp.close.diff()
    temp['open']=temp.open.diff()
    temp['high']=temp.high.diff()
    temp['low']=temp.low.diff()
    temp=temp.dropna()
    if len(temp)>n_steps:
        with tf.Session() as sess:
            saver.restore(sess, "lstm_model.ckpt")
            temp.clsoe.diff()
            ts=get_ts(standard(temp))
            predict=sess.run(pred,feed_dict={x: ts})  
        pre=convert(predict)
        context.function.predict({'预测状态':pre})   
    
def on_order(context):
	# my_file_name = 'test.file'
	# with open('myfile1', my_file_name) as f:
	#     pass
	# context.var.myvar += '通过上传文件的文件名定位文件，文件名需要一致'
	pass

def on_newday(context):
	pass
	# context.var.myvar += '当子夜过后，平台会调用这个函数'
