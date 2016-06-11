#!/usr/bin/env python

import os
import argparse
import numpy as np

file_prefix = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', default=128, type=int, required=True,
        help='size of test data (MB)')
parser.add_argument('-c', '--classes', default=2, type=int, required=True,
        help='number of classes (>2)')

args = parser.parse_args()
size = args.size
classes = args.classes

f = open('{}/data.txt'.format(file_prefix), 'w+')

# generate data
mb = 1024 * 1024
buf = 0
for i in range(0, size):
    buf += 1
    labels = np.random.randint(classes, size=mb)[:, np.newaxis]
    features = np.random.randint(10, size=(mb, 5))
    records = zip(labels, features)
    for r in records:
        feature_str = ""
        for j in range(0, len(r[1])):
            feature_str += '{}:{} '.format(j + 1, r[1][j])
        record_str = '{} {}\n'.format(r[0][0], feature_str)
        f.write(record_str)
    if buf == 1024:
        f.flush()
        buf = 0
    print "Generate {} MB data".format(i)


f.close()
