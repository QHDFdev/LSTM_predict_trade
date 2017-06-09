#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:11:15 2017

@author: wanjun
"""
#%%
#a=0.0005   #开仓的手续费
#b=0.0010 
#导入相关库
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
import numpy as np
#%%
#导入数据
#这个数据是用0填充了的数据
file_label='/Users/wanjun/Desktop/LSTM模型/code/data_big_change/data_big/label.csv'
file_data_big='/Users/wanjun/Desktop/LSTM模型/code/data_big_change/data_big/data_big.csv'
label=pd.read_csv(file_label,index_col=0,parse_dates=True)
data_big=pd.read_csv(file_data_big,index_col=0)
#%%
#定义函数，随机抓取batch_size个训练数据
#one_hot对label进行编码（这里是三类标签）
def get_train_data(data_big,label):
    lst=[]
    batch_x=[]
    batch_y=[]
    m=len(data_big.index.unique())
    lst=np.random.choice(range(m),size=batch_size,replace=True)
    #lst=np.random.randint(0,m,size=batch_size)
    for i in lst:
        batch_x.append(data_big.ix[i].values.tolist())
        #batch_x.append(data.ix[i:i+T].values.tolist())
        if label.ix[i]['label']==1:
            batch_y.append([1,0,0])
        elif label.ix[i]['label']==0:
            batch_y.append([0,1,0])
        else:
            batch_y.append([0,0,1])
    return batch_x,batch_y
def get_test_data(data_big,label,n_test):
    lst=[]
    batch_x=[]
    batch_y=[]
    m=len(data_big.index.unique())
    lst=np.random.choice(range(m),size=n_test,replace=True)
    for i in lst:
        batch_x.append(data_big.ix[i].values.tolist())
        #batch_x.append(data.ix[i:i+T].values.tolist())
        if label.ix[i]['label']==1:
            batch_y.append([1,0,0])
        elif label.ix[i]['label']==0:
            batch_y.append([0,1,0])
        else:
            batch_y.append([0,0,1])
    return batch_x,batch_y
#划分测试集与训练集
#data,a为划分的比列
def data_test_train(data_big,a):
    m=int(len(data_big.index.unique())*a)
    data_train=data_big.ix[:m]
    data_test=data_big.ix[m+1:]
    return data_train,data_test,m
def label_test_train(label,m):
    m+=n_steps
    label_train=label[:m]
    label_test=label[m:]
    return label_train,label_test
#%%
#定义参数
learning_rate = 0.001       #学习率
training_iters = 2000000     #最大迭代次数
batch_size =  500          #minibatch选择的大小
display_step = 10
#网络参数
n_input = 24 # 时间序列点的特征数
n_steps = 60 # 时间序列的长度（现在是最长的时间序列长度，不足的需要补齐为零）
n_hidden = 240  # 隐藏层结点数
n_classes = 3 # 标签数量
keep_prob=0.7 #随机选择神经元比例
m=2           #隐藏层的个数
a=0.9         #划分比列（训练集合测试集合）
erro_lst=[]     #损失函数的值
#划分训练集合和测试集合
#划分label
data_train,data_test,m=data_test_train(data_big,a)
label_train,label_test=label_test_train(label,m)
data_test=data_test.ix[m+n_steps:]
data_test.index-=(m+n_steps)
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

def real_len(x):
    #按照第三维度，全部等于零的记录为0,其余都是有正数的，所以记为1
    dense_sign = tf.sign(tf.reduce_max(tf.abs(x),reduction_indices=2))
    #全部相加，就能得到实际的长度
    length = tf.reduce_sum(input_tensor=dense_sign, reduction_indices=1)
    #转换函数，转换dtype
    length = tf.cast(length, tf.int32)
    #注意这里得到的length实际上是一个(batch_size)长度的向量
    return length
def cut_output(output,length):
    #length 输入时间序列的实际长度
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    #将每个batchsize的最后一个output所在的指标保存
    index = tf.range(start=0, limit=batch_size)*max_length + (length-1) 
    #先将输出展平，然后输出为x*out_put_size
    flat = tf.reshape(output, [-1,output_size]) 
    #根据实际长度来选择output
    result = tf.gather(flat, index) 
    return result

#由于是变长的时间序列，所以我们需要使用tf.nn.dynamic_rnn
def RNN(x, weights, biases):
    #这里GRU是另外一种加了门的RNN，可以看成是LSTM的变体
    layer=rnn.GRUCell(n_hidden)
    #layer=rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    #包裹dropout防止过拟和
    layer=rnn.DropoutWrapper(cell=layer,output_keep_prob=keep_prob)
    #最后一层不放dropout
    layer_out=rnn.GRUCell(n_hidden)
    #拼装成整体
    #这个地方写成列表的相加会陷入死循环，没有找到原因，所以就用最笨的方法全部写出来，如果要增加层数就直接增加layer
    layers=rnn.MultiRNNCell(cells=[layer,layer,layer_out])
    #计算实际长度
    length=real_len(x)
    #时间推进
    outputs,states=tf.nn.dynamic_rnn(cell=layers,inputs=x,dtype=tf.float32,sequence_length=length)
    #输出的outpus进行裁剪
    outputs=cut_output(outputs,length)
    # 输出函数使用的是线性函数
    # 时间序列的最后一个作为输出
    return tf.matmul(outputs, weights['out']) + biases['out']

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
    while step * batch_size < training_iters:
        batch_x, batch_y = get_train_data(data_train,label_train)     
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            erro_lst.append(loss)
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            erro_lst.append(loss)
        step += 1
    print("Optimization Finished!")
    #保存模型
    saver = tf.train.Saver()
    saver.save(sess,"model_ts_change/lstm_model.ckpt")
#%%
    #测试的数量
    n_test=len(data_test.index.unique())
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    test_data,test_label=get_test_data(data_test,label_test,n_test)
    print("Testing Accuracy:", \
       sess.run(accuracy, feed_dict={x: test_data, y: test_label}))