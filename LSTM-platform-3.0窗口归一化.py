#coding: UTF-8
# -*- coding: utf-8 -*-
"""

@author: wanjun
LSTM平台测试
回测版本

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
    #解压模型
    context.function.log('init')
    tar=tarfile.open('model_big.tar.gz')
    tar.extractall()
    tar.close()
    context.function.log('解压成功')

    #模型的参数 （如果参数改变，则训练的模型也要改变）
    #网络参数
    global n_input,n_steps,n_hidden,n_classes
    n_input = 24 # 时间序列点的特征数
    n_steps = 30 # 时间序列的长度
    n_hidden = 240 # 隐藏层结点数
    n_classes = 3 # 标签数量
    #每分钟的基础指标 
    context.var.ts=pd.DataFrame(columns=['open','close','high','low','volume','EMA_5min',
       'EMA_10min', 'EMA_15min', 'EMA_20min', 'RSI', 'upperband',
       'middleband', 'lowerband', 'macd', 'macdsignal', 'macdhist',
       'slowk', 'slowd', 'ATR', 'WILLR', 'OBV', 'SAR', 'DEMA', 'MOM'])
    context.var.pred=[]
    context.var.real=[]
    context.var.success=0
    context.var.sum=0
    context.var.fail=0
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
    #当天的真实数据
    context.var.history=None
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
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # 输出函数使用的是线性函数
    # 时间序列的最后一个作为输出
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
#计算指标
def technical_index(context):
    #EMA,RSI指标
    context.var.ts['EMA_5min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=5)
    context.var.ts['EMA_10min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=10)
    context.var.ts['EMA_15min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=15)
    context.var.ts['EMA_20min'] = talib.EMA(np.array(context.var.ts.close), timeperiod=20)
    context.var.ts['RSI'] = talib.RSI(np.array(context.var.ts.close))
    #STOCHRSI_usual = talib.STOCHRSI(np.array(context.var.ts.close))
    
    # BOLL-BAND指标
    BBANDS_usual = talib.BBANDS(np.array(context.var.ts.close))
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
    temp=temp.apply(lambda x: preprocessing.minmax_scale(x))
    return temp
#提取时间序列函数
def get_ts(temp):
    lst=[]
    lst.append(temp[-n_steps:].values.tolist())
    return lst
#转换预测值为标签
def convert(predict):
    n=np.argmax(predict)
    if n==0:
        return "可买涨"
    elif n==1:
        return "观望"
    else:
        return "可买跌"
#判断预测的实际效果
def real(context,p):
    lst=[]
    n=context.var.count
    ts=context.var.history.iloc[n:n+15]
    m=ts.iloc[0]['close']
    if p=='观望':
        return 0
    for i in ts['close']:
        change=i*(1-0.0010)-m*(1+0.0005)
        change1=m*(1-0.0005)-i*(1+0.0010)
        if change>0 and i>m:
            lst.append(1)
        elif change1>0 and i<m:
            lst.append(-1)
        else:
            lst.append(0)
    lst=np.array(lst)
    if (lst==1).sum()>0 and p=='可买涨':
        return 1
    elif (lst==-1).sum()>0 and p=='可买跌':
        return 1
    else:
        return -1
#%%       
def on_start(context):
    context.function.log(u'LSTM start')
    context.var.beginTrace = True
     #获取当前时刻的时间和交易日期
    datetime,date = context.function.get_time()
    #回测。获取当天历史数据
    ts=pd.DataFrame(columns=['open','close','high','low','volume'])
    df=pd.DataFrame(np.random.randn(1,5),columns=['open','close','high','low','volume'])
    data = context.function.get_market_data(datetime)
    for j in data:
        df.open=j['open']
        df.close=j['close']
        df.high=j['high']
        df.low=j['low']
        df.volume=j['volume']
        ts=ts.append(df,ignore_index=True)
    context.var.history=ts
    context.function.log('回测：成功获取当天历史数据')
def on_stop(context):
    context.function.log(u'LSTM stoped')
    context.var.beginTrace = False
def on_tick(context):
    pass
#%%  
def on_bar(context):
    if not context.var.beginTrace:
        return
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
    temp=context.var.ts.copy()
    temp=temp.dropna()
    context.function.log(context.var.count)
    if len(temp)>=n_steps:
        with tf.Session() as sess:
            saver.restore(sess, "model_big_2/lstm_model.ckpt")
            ts=get_ts(standard(temp))
            predict=sess.run(pred,feed_dict={x: ts})  
        p=convert(predict)
        accur=real(context,p)   #   预测正确返回1，预测错误返回-1 不关心的预测返回0
        context.var.pred.append(p)
        context.var.real.append(accur)
        lst=np.array(context.var.real)
        rate=(lst==1).sum()/float((lst!=0).sum())
        context.function.predict({'预测操作':p,'预测是否正确':accur,'正确率':rate})
    #计数
    context.var.count+=1   
def on_order(context):
	# my_file_name = 'test.file'
	# with open('myfile1', my_file_name) as f:
	#     pass
	# context.var.myvar += '通过上传文件的文件名定位文件，文件名需要一致'
	pass

def on_newday(context):
	    #清理每天的数据
    context.var.success=0
    context.var.sum=0
    context.var.fail=0
    if len(context.var.ts)>200:
        context.var.ts=context.var.ts[-200:]
    context.var.pred=[]
    context.var.real=[]
    context.var.count=0
    context.function.log(u'clear data')
     #获取当前时刻的时间和交易日期
    datetime,date = context.function.get_time()
    #回测。获取当天历史数据
    ts=pd.DataFrame(columns=['open','close','high','low','volume'])
    df=pd.DataFrame(np.random.randn(1,5),columns=['open','close','high','low','volume'])
    data = context.function.get_market_data(datetime)
    for j in data:
        df.open=j['open']
        df.close=j['close']
        df.high=j['high']
        df.low=j['low']
        df.volume=j['volume']
        ts=ts.append(df,ignore_index=True)
    context.var.history=ts
    context.function.log('回测：成功获取当天历史数据')
	# context.var.myvar += '当子夜过后，平台会调用这个函数'
