#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:00:31 2017

@author: wanjun
"""
#%%
#导入相关库
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
#%%
#导入数据
file_data='/Users/wanjun/LSTM/data.csv'
file_label='/Users/wanjun/LSTM/label.csv'
data=pd.read_csv(file_data,index_col='datetime',parse_dates=True)
label=pd.read_csv(file_label,index_col='datetime',parse_dates=True)

#%%
#定义函数，随机抓取batch_size个训练数据
#label_finl是data的标签数据
def get_train_data(data,label):
    T=n_steps
    lst=[]
    batch_x=[]
    batch_y=[]
    m=len(data)-T
    lst=np.random.choice(range(m),size=batch_size)
    #lst=np.random.randint(0,m,size=batch_size)
    for i in lst:
        batch_x.append(data.ix[i:i+T].values.tolist())
        if label.ix[i+T-1]['label']==1:
            batch_y.append([1,0,0])
        elif label.ix[i+T-1]['label']==0:
            batch_y.append([0,1,0])
        else:
            batch_y.append([0,0,1])
    return batch_x,batch_y
def get_test_data(data,label):
    T=n_steps
    lst=[]
    batch_x=[]
    batch_y=[]
    m=len(data)-T
    lst=range(m)
    for i in lst:
        batch_x.append(data.ix[i:i+T].values.tolist())
        if label.ix[i+T-1]['label']==1:
            batch_y.append([1,0,0])
        elif label.ix[i+T-1]['label']==0:
            batch_y.append([0,1,0])
        else:
            batch_y.append([0,0,1])
    return batch_x,batch_y
#划分测试集与训练集
#data,a为划分的比列
def data_test_train(data,a):
    m=int(len(data)*a)
    data_train=data[:m]
    data_test=data[m:]
    return data_train,data_test
def label_test_train(label,a):
    m=int(len(data)*a)
    label_train=label[:m]
    label_test=label[m:]
    return label_train,label_test
#%%
#定义参数
learning_rate = 0.003       #学习率
training_iters = 2000000    #最大迭代次数
batch_size =  500          #minibatch选择的大小
display_step = 10

#网络参数
n_input = 24 # 时间序列点的特征数
n_steps = 45 # 时间序列的长度
n_hidden = 72 # 隐藏层结点数
n_classes = 3 # 标签数量
a=0.8         #划分比列（训练集合测试集合）
num_layers=3  #cell的层数
keep_prob=0.8 #0,1之间的数字，越靠近1 dropout的东西就越少（正则化）
#划分训练集合和测试集合
#划分label
data_train,data_test=data_test_train(data,a)
label_train,label_test=label_test_train(label,a)
#%%
# 生成tf图
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 定义W,B 
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    cell =rnn.MultiRNNCell([lstm_cell] * num_layers)
    _initial_state=cell.zero_state(batch_size, tf.float32)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32,initial_state=_initial_state)
    # 输出函数使用的是线性函数
    # 时间序列的最后一个作为输出
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
#%%
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x, batch_y = get_train_data(data_train,label_train)
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    #保存模型
    saver = tf.train.Saver()
    saver.save(sess,"model-mul1/lstm_model.ckpt")
#%%
    #测试的数量
    n_test=len(data_test)
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    test_data,test_label=get_test_data(data_test,label_test)
    print("Testing Accuracy:", \
       sess.run(accuracy, feed_dict={x: test_data, y: test_label}))