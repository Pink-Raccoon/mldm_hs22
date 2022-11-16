# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:15:33 2022

@author: ashas
"""

import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston, make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
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
#   feature_names=cancer.feature_names, impurity=False, filled=True)
# with open("tree.dot") as f:
#   dot_graph = f.read()
# display(graphviz.Source(dot_graph))
# print("Feature importances:\n{}".format(tree.feature_importances_))



# def plot_feature_importances_cancer(model):
#  n_features = cancer.data.shape[1]
#  plt.barh(range(n_features), model.feature_importances_, align='center')
#  plt.yticks(np.arange(n_features), cancer.feature_names)
#  plt.xlabel("Feature importance")
#  plt.ylabel("Feature")
#  plt.ylim(-1, n_features)
# plot_feature_importances_cancer(tree)


ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,
 "ram_price.csv"))
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")

# use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]
# predict prices based on date
X_train = data_train.date[:, np.newaxis]
# we use a log-transform to get a simpler relationship of data to target
y_train = np.log(data_train.price)
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
# predict on all data
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)
# undo log-transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()