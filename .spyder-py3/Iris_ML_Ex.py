# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
'''
Imports required dependencies for the Iris Machine Learning example.
'''
iris_dataset = load_iris()

print("Target Names: {}".format(iris_dataset['target_names']))
print("Feature Names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
"""
Prints out information about the data we are using in our model.
"""
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))
"""
Prints out the targete data
"""
X_Train, X_Test, Y_Train, Y_Test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

print("X_Train shape: {}".format(X_Train.shape))
print("Y_Train shape: {}".format(Y_Train.shape))
print("X_Test shape: {}".format(X_Test.shape))
print("Y_Test shape: {}".format(Y_Test.shape))
"""
Creates the train and test data sets from our inital data set.
"""

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(X_Train, Y_Train)
"""
Uses the Kth Nearest Neighbors classifier with a single neighbor to train the model for fit.
"""
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape {}".format(X_new.shape))
"""
Creates a new observation to be used in our model.
"""

prediction = knn.predict(X_new)
print("Predicition: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
"""
Predicts the placement of the new observation.
"""
y_pred = knn.predict(X_Test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score (np.mean): {:.2f}".format(np.mean(y_pred == Y_Test)))
"""
Tests the accuracy of our model on the test data
"""
