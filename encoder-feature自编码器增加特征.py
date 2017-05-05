#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:17:12 2017

@author: wanjun
encode 增加特征
"""
#自定义函数
#随机抓取时间序列
def random_data_ts(data,n):
    m=len(data)-T
    choice=np.random.choice(range(m),size=n)
    lst=[]
    for i in choice:
        lst.append(data[i:i+T].reshape(5*T))
    return lst
#随机抓取时间点
def random_data_dot(data,n):
    lst=[]
    m=len(data)
    choice=np.random.choice(range(m),size=n)
    for i in choice:
        lst.append(data[i])
    return lst
    
#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import talib 
#%%
# 导入原始数据数据
file='/Users/wanjun/Desktop/深度学习与神经网络/data/data_train.csv'
data=pd.read_csv(file,index_col='datetime',parse_dates=True)
#数据处理：算差加上归一化,volume已经算过，所以不算差
data['open']=data.open.diff()
data['high']=data.high.diff()
data['low']=data.low.diff()
data['close']=data.close.diff()
data=data[1:]
_index=data.index
data=preprocessing.minmax_scale(data)
#%%
# 参数
learning_rate = 0.001    #学习速率
training_epochs = 100    #训练批次
batch_size = 5000       #随机选择训练数据大小
display_step = 1        #展示步骤
T=20                    #时间序列的长度 
erro_lst=[]             #误差曲线
#%%
# 网络参数
n_hidden_1 = 10  #第一隐层神经元数量
n_hidden_2 = 20  #第二
n_hidden_3 = 32  #第三
n_input = 5     #输入:时间序列就是5*T ,点就是5
#%%
#tf Graph输入
X = tf.placeholder("float", [None,n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}

#偏置值初始化
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input])),
}

def encoder(x): 
    #sigmoid激活函数，layer = x*weights['encoder_h1']+biases['encoder_b1']
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),biases['encoder_b3']))
    return layer_3

# 开始解码
def decoder(x):
    #sigmoid激活函数,layer = x*weights['decoder_h1']+biases['decoder_b1']
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),biases['decoder_b3']))
    return layer_3

#%%
# 构造模型
encoder_op = encoder(X)
encoder_result = encoder_op
decoder_op = decoder(encoder_op)

#编码后的
y_pred = decoder_op
#实际输入数据
y_true = X

# 定义代价函数和优化器，最小化平方误差,这里可以根据实际修改误差模型
cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 初始化变量
init=tf.global_variables_initializer()

#%%
# 运行Graph
with tf.Session() as sess:
    sess.run(init)
    #训练的组数
    total_batch = int(len(data)/batch_size)
    # 开始训练
    for epoch in range(training_epochs):    #训练的次数
        for i in range(total_batch):        #每次训练的组数
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #batch_xs=data_lst[i*batch_size:i*batch_size+batch_size]
            batch_xs=random_data_dot(data,batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            erro_lst.append(c)
    # 展示每次训练结果
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))   
    print("Optimization Finished!")
    #存储模型数据
    saver=tf.train.Saver()
    saver.save(sess,"encoder-mdoel/encoder.ckpt")
    #计算编码后的值
    encodes =sess.run(encoder_result,feed_dict={X: data})

#%%
encodes=pd.DataFrame(encodes,index=_index,columns=range(n_hidden_3))
encodes=encodes.apply(lambda x:preprocessing.minmax_scale(x))
encodes.to_csv('encodes_feature')     
#%%
