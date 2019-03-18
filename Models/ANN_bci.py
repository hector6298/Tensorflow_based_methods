import mne
import numpy as np
import pandas as pd
import matplotlib 
#import matplotlib.pyplot as plt
import scipy.io
import sklearn
import tensorflow as tf
from tensorflow.contrib.signal.python.ops import window_ops
import functools
import Tensorflow_neural_methods as tfnm
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.io
from numpy import genfromtxt
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn import datasets

import LeerDatSav as lSav

LOGDIR = 'tmp/eeg4/dustin/'
path = 'RWE_dustin/Test 3/Experiment_1/level_4/folds/'
files = [['fold_'+str(i)+'_train'+'.sav','fold_'+str(i)+'_test'+'.sav'] for i in range(1,11)]
onehot_encoder = OneHotEncoder(sparse=False)

#### best hyperparameters ######## :v :v :v :v 
ns = [106]
bn = True
dropout = False
use_l2 = False
lr = 0.01
epochs = 500
sess=tfnm.setupGPU()

#######################################
predictions_per_fold = []
round_pred_per_fold = []
real_labels_per_fold = []
#######################################

X = tf.placeholder(tf.float32,shape = [None,160])
Y = tf.placeholder(tf.float32,shape=[None,32])
BATCH_SIZE = tf.placeholder(tf.int64)
x,y,iterator = tfnm.input_fn(X,Y,BATCH_SIZE,tf.float32,shuffle_repeat=False)
is_train = tf.placeholder(tf.bool)

################ graph ###################################
fcl,l2 = tfnm.fc_layers(x,ns,bn=bn,use_dropout = dropout,is_train=is_train,use_l2=use_l2)
logits,l2_ = tfnm.fullyCon(fcl,32,batch_norm=bn,is_train=is_train,l2=use_l2)

l2_loss = l2+l2_
scaled  = tf.nn.softmax(logits,name='softmax')
loss = tfnm.loss_func_multi(logits,y)

accuracy,acc_summary = tfnm.define_accuracy_multi(scaled,y)
optimizer = tf.train.AdamOptimizer
train_step =tfnm.define_train_step(optimizer,lr,loss)

summ = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())


trainwriter = tf.summary.FileWriter(LOGDIR+'train')
trainwriter.add_graph(sess.graph)
testwriter = tf.summary.FileWriter(LOGDIR+'test')
testwriter.add_graph(sess.graph)

validations = []
#################################### DATA ITERATION AND TRAINNING COMPUTATION ####################################
for fold_train,fold_test in files:
    sess.run(tf.global_variables_initializer())
    data_train = np.array(lSav.readSav(path+fold_train)['data'])
    labels_train = np.array(lSav.readSav(path+fold_train)['labels'])
    data_test = np.array(lSav.readSav(path+fold_test)['data'])
    labels_test = np.array(lSav.readSav(path+fold_test)['labels'])


    labels_train = labels_train.reshape(len(labels_train), 1)
    labels_test = labels_test.reshape(len(labels_test),1)
    oh_labels_train = onehot_encoder.fit_transform(labels_train)
    oh_labels_test = onehot_encoder.fit_transform(labels_test)
    print(oh_labels_train)


    sess.run(iterator.initializer,feed_dict={X:data_train,Y:oh_labels_train,BATCH_SIZE:100,is_train:True})


    for i in range(int(epochs*data_train.shape[0]/BATCH_SIZE.eval(feed_dict={BATCH_SIZE:100}))):
        #train
        if i%100 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ],feed_dict={is_train:True})
            trainwriter.add_summary(s,i)
            print("step %d, training accuracy %g"%(i, float(train_accuracy)))
        train_step.run(feed_dict={is_train:True})

    #test

    sess.run(iterator.initializer,feed_dict={X:data_test,Y:oh_labels_test,BATCH_SIZE:oh_labels_test.shape[0],is_train:False})
    summary, val_acc = sess.run([acc_summary,accuracy],feed_dict={is_train:False})
    pred = scaled.eval(feed_dict = {is_train:False})
    predictions_per_fold.append(np.array(pred))
    round_pred_per_fold.append(np.array(np.argmax(pred)))
    real_labels_per_fold.append(np.array(oh_labels_test))
    testwriter.add_summary(summary,0)
    print("Validation accuracy %g"%val_acc) 
    validations.append(val_acc)

validations = np.array(validations)
mean_val = np.mean(validations)

sess.close()
