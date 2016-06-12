#!/usr/bin/env python

import os
import numpy as np
import time
from sklearn import svm

file_prefix = os.getcwd()

f = open('{}/local_data.txt'.format(file_prefix))
matrix = np.asarray([ map(float,line.split(' ')) for line in f ])
print matrix.shape[0]
number = matrix.shape[0]
train_size = number * 0.7

training = matrix[:train_size][:]
validation = matrix[train_size:][:]

training_label = training[:, 0]
training_feature = training[:, 1:]

clf = svm.SVC(kernel='sigmoid')
print "start training"
start_time = time.time()
clf.fit(training_feature, training_label)
delta = time.time() - start_time
print "finish traning with {}s".format(delta)