import Tensorflow_neural_methods as tfnm
import numpy as np
import tensorflow as tf
from numpy import argmax
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.io
from numpy import genfromtxt
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn import datasets

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
mode =2
sess = tfnm.setupGPU()

LOGDIR = '/tmp/EEG/'

path = 'EEG/'
files_label = ["label.csv","label8.csv","label9.csv","label10.csv","label11.csv","label12.csv","label14.csv","label15.csv","label16.csv","label18.csv","label19.csv","label20.csv","label21.csv"]
files_data = ["mfcc.csv","mfcc8.csv","mfcc9.csv","mfcc10.csv","mfcc11.csv","mfcc12.csv","mfcc14.csv","mfcc15.csv","mfcc16.csv","mfcc18.csv","mfcc19.csv","mfcc20.csv","mfcc21.csv"]


data = scipy.io.loadmat('EEG/'+'MM05/'+'all_features_ICA.mat')['all_features']['eeg_features'][0][0][0]['thinking_feats'][0][0]
Data = np.array([data[i] for i in range(165)])
data = Data
#prompts = scipy.io.loadmat('all_features_ICA.mat')['all_features']['prompts'][0][0][0]
#for i in range(len(files_data)-1):
labels = np.loadtxt('mfcc/'+files_label[0],delimiter=',',dtype=str)
#my_labels = np.append(my_labels,labels)
shapeX = data[0].shape[1]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

X = tf.placeholder(tf.float32,shape = [None,64,1197], name = 'x')
Y = tf.placeholder(tf.float32,shape=[None,11], name = 'labels')
is_train = tf.placeholder(tf.bool,name='is_train')
BATCH_SIZE = tf.placeholder(tf.int64)

x,y,iterator = tfnm.input_fn(X,Y,BATCH_SIZE,tf.float32)

##c1 =  tfnm.conv_layer1D(x,5,64,128,name = 'conv1D_1',pname='MP1D_1')
#c2 = tfnm.conv_layer1D(c1,4,128,256,name = 'conv1D_2',pname='MP1D_2')

flatten = tf.reshape(x,[-1,64*shapeX])

fcl = tfnm.fc_layers(flatten,[100,100,50,25],use_dropout=True)
logits = tfnm.fullyCon(fcl,11)

loss = tfnm.loss_func_multi(logits,y)

accuracy,acc_summary = tfnm.define_accuracy_multi(logits,y)
optimizer = tf.train.AdamOptimizer
train_step =tfnm.define_train_step(optimizer,0.01,loss)

summ = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())


sess.run(tf.global_variables_initializer())
trainwriter = tf.summary.FileWriter(LOGDIR+'train')
trainwriter.add_graph(sess.graph)
testwriter = tf.summary.FileWriter(LOGDIR+'test')
testwriter.add_graph(sess.graph)

sess.run(iterator.initializer,feed_dict={X:data,Y:onehot_encoded,BATCH_SIZE:30,is_train:True})


for i in range(100000):
    #train
    if i%100 == 0:
        [train_accuracy, s] = sess.run([accuracy, summ],feed_dict={is_train:True})
        trainwriter.add_summary(s,i)
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={is_train:True})

#test

sess.run(iterator.initializer,feed_dict={X:data,Y:onehot_encoded,BATCH_SIZE:data.shape[0],is_train:False})
summary, val_acc = sess.run([acc_summary,accuracy],feed_dict={is_train:False})
testwriter.add_summary(summary,0)
print("Validation accuracy %g"%val_acc) 




