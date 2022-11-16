# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:48:13 2022

@author: Asha
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import mglearn

knn = KNeighborsClassifier(n_neighbors=1)


iris_dataset = load_iris()
x = iris_dataset['data']
y = iris_dataset['target']

print (f"keys of iris_dataset: \n {iris_dataset.keys()}")
print (f"keys of iris_dataset: \n {iris_dataset['feature_names']}")

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print(f"X_train shape: {X_train.shape}")

iris_df = pd.DataFrame(X_train, columns = iris_dataset.feature_names)

pd.plotting.scatter_matrix(iris_df,c=y_train, 
                           figsize=(15,15),marker='0', 
                           hist_kwds={'bins':20}, s=60, 
                           alpha=0.8, cmap=mglearn.cm3)


knn.fit(X_train, y_train)

X_new = np.array([[5,2.9,1,0.2]])
pred = knn.predict(X_new)
y_pred = knn.predict(X_test)

print(f"Predicted target name: {iris_dataset['target_names'][y_pred]}")