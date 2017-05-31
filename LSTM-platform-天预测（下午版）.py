#coding: UTF-8
# -*- coding: utf-8 -*-
"""

@author: wanjun
LSTM平台测试
预测当天上午的涨幅情况

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
from datetime import timedelta
from collections import defaultdict

#%%
def on_init(context):    
    #解压模型
    context.function.log('init')
    tar=tarfile.open('model.tar.gz')
    tar.extractall()
    tar.close()
    context.function.log('解压成功')

    #模型的参数 （如果参数改变，则训练的模型也要改变）
    #网络参数
    global n_input,n_steps,n_hidden,n_classes
    n_input = 5 # 时间序列点的特征数
    n_steps = 45 # 时间序列的长度
    n_hidden = 300 # 隐藏层结点数
    n_classes = 3 # 标签数量
    #每分钟的基础指标 
    context.var.ts=pd.DataFrame(columns=['open','close','high','low','volume'])
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
#%%
#自定义函数
#生成lstm    
def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
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
        return "涨"
    elif n==1:
        return "平"
    else:
        return "跌"
#%%       
def on_start(context):
    context.function.log(u'LSTM start')
    context.var.beginTrace = True
def on_stop(context):
    context.function.log(u'LSTM stoped')
    context.var.beginTrace = False
def on_tick(context):
    pass
#%%  
def on_bar(context):
    if not context.var.beginTrace:
        return
    #计数
    context.var.count+=1  
    #获取当前时刻的bar数据
    if context.var.count<=90 or context.var.count>=150:
        return 
    bar=context.var.bar
    #组合得到新的ts
    df=pd.DataFrame(np.random.randn(1,5),columns=context.var.ts.columns)
    df['close']=bar.close
    df['open']=bar.open
    df['low']=bar.low
    df['high']=bar.high
    df['volume']=bar.volume
    context.var.ts=context.var.ts.append(df,ignore_index=True)
    temp=context.var.ts.copy()
    context.function.log(context.var.count)
    context.function.log(len(temp))
    if len(temp)==n_steps:
        with tf.Session() as sess:
            saver.restore(sess, "model/lstm_model.ckpt")
            ts=get_ts(standard(temp))
            predict=sess.run(pred,feed_dict={x: ts})  
        p=convert(predict)
        context.function.predict({'今日下午走势':p})
     
def on_order(context):
	# my_file_name = 'test.file'
	# with open('myfile1', my_file_name) as f:
	#     pass
	# context.var.myvar += '通过上传文件的文件名定位文件，文件名需要一致'
	pass

def on_newday(context):
	    #清理每天的数据
    context.var.ts=pd.DataFrame(columns=['open','close','high','low','volume'])
    context.var.count=0
    context.function.log(u'clear data')
	# context.var.myvar += '当子夜过后，平台会调用这个函数'
