#!/usr/bin/env python

import os
import numpy as np
import time
from sklearn import svm
from sklearn import metrics

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

clf = svm.SVC(kernel='rbf', gamma=0.7, C=1)
print "start training"
start_time = time.time()
clf.fit(training_feature, training_label)
delta = time.time() - start_time
print "finish traning with {}s".format(delta)


# start evaluating the precision
print "start evaluation"
valid_labels = validation[:, 0]
valid_features = validation[:, 1:]
res = clf.predict(valid_features)
precision = metrics.precision_score(valid_labels, res, average="binary")
recall = metrics.recall_score(valid_labels, res, average="binary")
print "precision: {}, recall: {}".format(precision, recall)
