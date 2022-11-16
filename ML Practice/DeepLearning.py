# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:48:58 2022

@author: Asha
"""

import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC



line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")