# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:15:33 2022

@author: ashas
"""

import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import graphviz

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

# export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
#  feature_names=cancer.feature_names, impurity=False, filled=True)
# with open("tree.dot") as f:
#  dot_graph = f.read()
# graphviz.Source(dot_graph).view()
print("Feature importances:\n{}".format(tree.feature_importances_))
def plot_feature_importances_cancer(model):
 n_features = cancer.data.shape[1]
 plt.barh(range(n_features), model.feature_importances_, align='center')
 plt.yticks(np.arange(n_features), cancer.feature_names)
 plt.xlabel("Feature importance")
 plt.ylabel("Feature")
 plt.ylim(-1, n_features)
plot_feature_importances_cancer(tree)