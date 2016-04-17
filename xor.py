# coding: utf-8
'''
    Tensorflow - XOR
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 教師データ
train_x = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]], dtype = np.float32)
train_y = np.array([[0.0],[1.0],[1.0],[0.0]],dtype = np.float32)

# プレースホルダ
_x = tf.placeholder(tf.float32, shape=[None,2])
_y = tf.placeholder(tf.float32, shape=[None,1])

# 式定義
w1 = tf.Variable(tf.random_uniform([2,3], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([1,3]))
w2 = tf.Variable(tf.random_uniform([3,1], -1.0, 1.0))
b2 = tf.Variable(tf.zeros([1,1]))

# 隠れ層(sigmoid)
hidden = tf.nn.sigmoid(tf.matmul(_x, w1) + b1)
y      = tf.nn.sigmoid(tf.matmul( hidden , w2) + b2)

# 誤差
cross_entropy = -(_y * tf.log(y) + (1 - _y) * tf.log(1 - y))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

# 計算の前準備
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
feed_dict =  {_x : train_x, _y : train_y}

# 計算
for step in range(0,2001) :
    sess.run(train, feed_dict = feed_dict)

    if step % 500 == 0 :
        print "step : {%d} " % step
        a_w1 = sess.run(w1, feed_dict = feed_dict)
        a_w2 = sess.run(w2, feed_dict = feed_dict)
        a_b1 = sess.run(b1, feed_dict = feed_dict)
        a_b2 = sess.run(b2, feed_dict = feed_dict)

        x1  = np.float32(np.array([[0.0,0.0]]))
        hidden = tf.nn.sigmoid(tf.matmul(x1, w1) + b1)
        y      = tf.nn.sigmoid(tf.matmul( hidden , w2) + b2)
        print "y(0,0):" , sess.run(y)

        x2  = np.float32(np.array([[0.0,1.0]]))
        hidden = tf.nn.sigmoid(tf.matmul(x2, w1) + b1)
        y      = tf.nn.sigmoid(tf.matmul( hidden , w2) + b2)
        print "y(0,1):" , sess.run(y)

        x3  = np.float32(np.array([[1.0,0.0]]))
        hidden = tf.nn.sigmoid(tf.matmul(x3, w1) + b1)
        y      = tf.nn.sigmoid(tf.matmul( hidden , w2) + b2)
        print "y(1,0):" , sess.run(y)

        x4  = np.float32(np.array([[1.0,1.0]]))
        hidden = tf.nn.sigmoid(tf.matmul(x4, w1) + b1)
        y      = tf.nn.sigmoid(tf.matmul( hidden , w2) + b2)
        print "y(1,1):" , sess.run(y)
