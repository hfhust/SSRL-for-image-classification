# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:41:24 2016

@author: ldy
"""



from time import time
from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn import svm
#
import numpy as np

from sklearn import metrics

import os
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# import tensorflow as tf

n=5
for i in range(n):
    acc = []

    nums = [90, 95,100,105,110,115,120]
    #100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175

    for num in nums:
        X_train = np.load('features%d/features%d_train.npy' %(i+1, num))
        y_train = np.load('features%d/label%d_train.npy' %(i+1, num))
        X_test = np.load('features%d/features%d_test.npy'%(i+1, num))
        y_test = np.load('features%d/label%d_test.npy'%(i+1, num))
        # print(X_train.shape)
        # print(X_train)
        print("Fitting the classifier to the training set")
        t0 = time()
        C = 1000.0  # SVM regularization parameter
        clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
        print("done in %0.3fs" % (time() - t0))

        print("Predicting...num is %d"%num)
        t0 = time()
        y_pred = clf.predict(X_test)
        print("done in %0.3fs" % (time() - t0))

        print("Accuracy: %.3f" % (accuracy_score(y_test, y_pred)))
        acc.append(accuracy_score(y_test, y_pred))
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        sys.stdout.flush()
    print(acc)
    print(confusion_matrix)




