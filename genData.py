#!/usr/bin/env python

import os
import argparse
import numpy as np
from sklearn import datasets

file_prefix = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', default=128, type=int, required=True,
        help='size of test data (rows)')
parser.add_argument('-c', '--classes', default=2, type=int, required=True,
        help='number of classes (>2)')
parser.add_argument('-f', '--features', default=2, type=int, required=True,
        help='number of features (>2)')


args = parser.parse_args()
size = args.size
classes = args.classes
features = args.features

f = open('{}/local_data.txt'.format(file_prefix), 'w+')

# generate data
x, y = datasets.make_classification(n_samples=size, n_features=features, n_classes=classes)

records = zip(y, x)
for r in records:
    r_str = str(r[0])
    for feature in r[1]:
        r_str += ' {}'.format(str(feature))
    r_str += '\n'
    f.write(r_str)

# mb = 1024 * 1024
# buf = 0
# for i in range(0, size):
#     buf += 1
#     labels = np.random.randint(classes, size=mb)[:, np.newaxis]
#     features = np.random.randint(1000, size=(mb, features))
#     records = zip(labels, features)
#     for r in records:
#         feature_str = ""
#         for j in range(0, len(r[1])):
#             feature_str += '{} '.format(r[1][j])
#         record_str = '{} {}\n'.format(r[0][0], feature_str)
#         f.write(record_str)
#     if buf == 1024:
#         f.flush()
#         buf = 0
#     print "Generate {} MB data".format(i)


f.close()
