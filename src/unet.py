import time
import os

import numpy as np

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d

import tensorflow as tf

from dataloader import DataSet
data = DataSet()

from helpers import (dice_score)

from config import (batch_size, IMG_WIDTH, 
            IMG_HEIGHT, IMG_CHANNELS, learning_rate)

# Network Parameters
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])
y = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS])

################Create Model######################
conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
pool1 = max_pool_2d(conv1, 2)

conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
pool2 = max_pool_2d(conv2, 2)

conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
pool3 = max_pool_2d(conv3, 2)

conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
pool4 = max_pool_2d(conv4, 2)

conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

up6 = upsample_2d(conv5,2)
up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

up7 = upsample_2d(conv6,2)
up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

up8 = upsample_2d(conv7,2)
up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

up9 = upsample_2d(conv8,2)
up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

pred = conv_2d(conv9, 1, 1,  activation='linear', padding='valid')
pred = tf.nn.softmax(pred)

cost = 1 - dice_score(pred, y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

###############Initialize Model#######################
merged = tf.summary.merge_all()
init = tf.initialize_all_variables()
    
sess = tf.Session()
saver = tf.train.Saver()
##############Evaluate Model on data#############
step = 0
with tf.Session() as sess:
	sess.run(init)
	train_writer = tf.summary.FileWriter('./logs', sess.graph)
	batch = data.train_batch(batch_size)
	feed_dict = {x: batch['x'], y: batch['y']} 
	summary, out, loss, _ = sess.run([merged, pred, cost, optimizer], feed_dict=feed_dict)
	train_writer.add_summary(summary, step)

	print "Batch Step= {:.1f}".format(step)+", loss= {:.6f}".format(loss)
	step += 1