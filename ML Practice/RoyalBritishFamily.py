# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:14:33 2022

@author: Asha
"""

import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, axes3d
from sklearn.svm import SVC



from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# X, y = make_blobs(centers=4, random_state=8)
# y = y % 2
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")


# linear_svm = LinearSVC().fit(X, y)
# mglearn.plots.plot_2d_separator(linear_svm, X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")


# X_new = np.hstack([X, X[:, 1:] ** 2])
# figure = plt.figure()
# visualize in 3D
# ax = Axes3D(figure, elev=-152, azim=-26)
# plot first all the points with y == 0, then all with y == 1

# coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# show linear decision boundary
# figure = plt.figure()
# ax = Axes3D(figure, elev=-152, azim=-26)
# xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
# yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
# XX, YY = np.meshgrid(xx, yy)
# # ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
# ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
# ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
#  cmap=mglearn.cm2, s=60, edgecolor='k')
# ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
#  cmap=mglearn.cm2, s=60, edgecolor='k')
# ax.set_xlabel("feature0")
# ax.set_ylabel("feature1")
# ax.set_zlabel("feature1 ** 2")


# ZZ = YY ** 2
# dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
# plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
#  cmap=mglearn.cm2, alpha=0.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")


# X, y = mglearn.tools.make_handcrafted_dataset()
# svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
# mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # plot support vectors
# sv = svm.support_vectors_
# # class labels of support vectors are given by the sign of the dual coefficients
# sv_labels = svm.dual_coef_.ravel() > 0
# mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")

X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, random_state=0)
svc = SVC()
svc.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
plt.boxplot(X_train)
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
