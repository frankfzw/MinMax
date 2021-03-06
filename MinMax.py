#!/usr/bin/env python

import os
import numpy as np
import time
import threading
import sys
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

positive = filter(lambda x: x[0] < 1, training)
negative = filter(lambda x: x[0] > 0, training)

pos_len = len(positive) / 4
neg_len = len(negative) / 4


print "start training"
start_time = time.time()
models = [None] * 16
thread_pool = []

def train_model(i, j, features, labels):
  print "start model {}:{} training".format(i, j)
  start_time = time.time()
  clf = svm.SVC(kernel='rbf', gamma=0.7, C=1)
  clf.fit(features, labels)
  models[i * 4 + j] = clf
  delta = time.time() - start_time
  print "finish model {}:{} training with {}s".format(i, j, delta)


for i in range(0, 4):
  for j in range(0, 4):
    tmp = np.concatenate((positive[(i * pos_len):((i + 1) * pos_len)], negative[(j * neg_len):((j + 1) * neg_len)]), axis=0)
    np.random.shuffle(tmp)
    labels = tmp[:, 0]
    features = tmp[:, 1:]
    try:
      t = threading.Thread(target=train_model, args=(i, j, features, labels, ))
      thread_pool.append(t)
      t.start()
    except:
      print "Error"
      sys.exit(1)

for t in thread_pool:
  t.join()

delta = time.time() - start_time
print "finish training with {}s".format(delta)

# start evaluating the precision
print "start evaluation"
valid_labels = validation[:, 0]
valid_features = validation[:, 1:]

def predict(fs):
  number = len(fs)
  res = []
  tmp_max = []
  for i in range(0, 4):
    tmp = []
    for j in range(0, 4):
      tmp.append(models[i * 4 + j].predict(fs))
    tmp_array = np.asarray(tmp)
    tmp_min = []
    for j in range(0, number):
      tmp_min.append(min(tmp_array[:, j]))
    tmp_max.append(tmp_min)

  tmp_max_array = np.asarray(tmp_max)
  for i in range(0, number):
    res.append(max(tmp_max_array[:, i]))

  return res


res = np.asarray(predict(valid_features))
print valid_labels
print res
precision = metrics.precision_score(valid_labels, res, average="binary")
recall = metrics.recall_score(valid_labels, res, average="binary")
print "precision: {}, recall: {}".format(precision, recall)


