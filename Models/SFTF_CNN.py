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





########data load#####################
path = "EEG/"
folders = ["MM05/","MM08/","MM09/","MM10/","MM11/","MM12/","MM14/","MM15/","MM16/","MM18","MM19/","MM20/","MM21/"]
filename = 'epoch_data.mat'
mat = scipy.io.loadmat(path+folders[1]+filename)['epoch_data']['thinking_mats']
data = mat[0][0][0] 
data = np.delete(data,(85,91))
sample_rate = 1000 #1kHz
window_size_ms = 25
window_stride_ms = 10

sess = tf.Session()
ds = []
for i in data:
    t = tf.convert_to_tensor(i,dtype = tf.float32)
    segment_size = i.shape[1]
    segment_size_samples = int(sample_rate * segment_size / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)

    #signals = tf.placeholder(tf.float32,shape= [62,segment_size_samples])
    stfts = tf.contrib.signal.stft(t,
                                frame_length=window_size_samples,
                                frame_step=window_stride_samples,
                                fft_length=window_size_samples,
                                window_fn=functools.partial(window_ops.hann_window, periodic=True))
    power_spectrograms = tf.real(stfts * tf.conj(stfts))
    magnitude_spectrograms = tf.abs(stfts)
    ds.append(magnitude_spectrograms)
ds = np.array(ds)
dataset = []
for i in ds:
    d = sess.run(i)
    dataset.append(d)
dataset = np.array(dataset)
print(dataset.shape)

sess.close()

############################### END OF SPECTROGRAM COMPUTATIONS  #################################################
labels = np.loadtxt('mfcc/label8.csv',delimiter=',',dtype=str)
labels = np.delete(labels,(85,91))


sess = tfnm.setupGPU()
LOGDIR = '/tmp/EEG/convolutional'

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

############### split data for train/test ##############################################################
train = int(labels.shape[0]*0.8)
test = labels.shape[0] - train
data_train = np.array([dataset[i] for i in range(train)])
labels_train = np.array([onehot_encoded[i] for i in range(train)])

data_test = np.array([dataset[i] for i in range(train,labels.shape[0])])
labels_test = np.array([onehot_encoded[i] for i in range(train,labels.shape[0])])

####################### graph ###############################
X = tf.placeholder(tf.float32,shape = [None,62,498,13], name = 'x')
Y = tf.placeholder(tf.float32,shape=[None,11], name = 'labels')
is_train = tf.placeholder(tf.bool,name='is_train')
BATCH_SIZE = tf.placeholder(tf.int64)


x,y,iterator = tfnm.input_fn(X,Y,BATCH_SIZE,tf.float32)

c1 =  tfnm.conv_layer2D(x,62,124,batch_norm= True,is_train = is_train,name = 'conv1D_1',pname='MP1D_1',bn_name='batch_normC1')
c2 = tfnm.conv_layer2D(c1,124,248,batch_norm = True,is_train=is_train,name = 'conv1D_2',pname='MP1D_2',bn_name='batch_normC2')

flatten = tf.reshape(c2,[-1,int(248*np.ceil(498/4)*np.ceil(13/4))])

fcl = tfnm.fc_layers(flatten,[100,100,50,25],bn=True,is_train = is_train,use_dropout=True)
logits = tfnm.fullyCon(fcl[0],11,batch_norm=True,is_train = is_train)
scaled  = tf.nn.softmax(logits,name='softmax')
loss = tfnm.loss_func_multi(logits,y)

accuracy,acc_summary = tfnm.define_accuracy_multi(scaled,y)
optimizer = tf.train.AdamOptimizer
train_step =tfnm.define_train_step(optimizer,0.01,loss)

summ = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())


sess.run(tf.global_variables_initializer())
trainwriter = tf.summary.FileWriter(LOGDIR+'train')
trainwriter.add_graph(sess.graph)
testwriter = tf.summary.FileWriter(LOGDIR+'test')
testwriter.add_graph(sess.graph)

sess.run(iterator.initializer,feed_dict={X:data_train,Y:labels_train,BATCH_SIZE:30,is_train:True})


for i in range(500):
    #train
    if i%100 == 0:
        [train_accuracy, s] = sess.run([accuracy, summ],feed_dict={is_train:True})
        trainwriter.add_summary(s,i)
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={is_train:True})

#test

sess.run(iterator.initializer,feed_dict={X:data_test,Y:labels_test,BATCH_SIZE:labels_test.shape[0],is_train:False})
summary, val_acc = sess.run([acc_summary,accuracy],feed_dict={is_train:False})
testwriter.add_summary(summary,0)
print("Validation accuracy %g"%val_acc) 












