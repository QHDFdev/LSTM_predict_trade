#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:11:15 2017

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
file_data='/Users/wanjun/Desktop/time/code/data_afternoon.csv'
file_label='/Users/wanjun/Desktop/time/code/label_afternoon.csv'
data=pd.read_csv(file_data,index_col=0,parse_dates=True)
label=pd.read_csv(file_label,index_col=0,parse_dates=True)
#label=label[1:]
#%%
#定义函数，随机抓取batch_size个训练数据
#label_finl是data的标签数据
def get_train_data(data,label):
    day=label.index
    batch_x=[]
    batch_y=[]
    for i in day:
        i=str(i)[:10]
        batch_x.append(data.ix[i].values.tolist())
        if label.ix[i].label==1:
            batch_y.append([1,0,0])
        elif label.ix[i].label==0:
            batch_y.append([0,1,0])
        else:
            batch_y.append([0,0,1])
    return batch_x,batch_y
def get_test_data(data,label,n_test):
    day=label.index
    batch_x=[]
    batch_y=[]
    for i in day:
        i=str(i)[:10]
        batch_x.append(data.ix[i].values.tolist())
        if label.ix[i].label==1:
            batch_y.append([1,0,0])
        elif label.ix[i].label==0:
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
learning_rate = 0.001       #学习率
training_iters = 600000  #最大迭代次数
batch_size =  500          #minibatch选择的大小
display_step = 10

#网络参数
n_input = 5 # 时间序列点的特征数
n_steps = 45 # 时间序列的长度
n_hidden = 300  # 隐藏层结点数
n_classes = 3 # 标签数量
erro_lst=[]     #损失函数的值
#划分训练集合和测试集合
data_train=data
data_test=data.ix['2017-01-03':]
label_train=label
label_test=label.ix['2017-01-03':]
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

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
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
            erro_lst.append(loss)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    #保存模型
    saver = tf.train.Saver()
    saver.save(sess,"model_after1/lstm_model.ckpt")
#%%
    #测试的数量
    n_test=len(data_test)
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    test_data,test_label=get_test_data(data_test,label_test,n_test)
    print("Testing Accuracy:", \
       sess.run(accuracy, feed_dict={x: test_data, y: test_label}))