from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import math
# from sklearn.neighbors import DistanceMetric

f = np.loadtxt(
    open(".\\src\\train.csv", "rb"), delimiter=",", skiprows=0)
train_data = f[:, :-1]
train_target = f[:, -1]
f = np.loadtxt(
    open(".\\src\\test.csv", "rb"), delimiter=",", skiprows=0)
test_data = f[:, :-1]
test_target = f[:, -1]

min_error = 1
best_n_neighbors = 0
for i in range(1, 201, 5):
    clf = KNeighborsClassifier(n_neighbors=i, metric='manhattan', p=1)
    clf.fit(train_data, train_target)
    test_accuracy = 1-clf.score(test_data, test_target)
    if test_accuracy < min_error:
        min_error = test_accuracy
        best_n_neighbors = i
print("Test KNN using Manhattan Distance min error rate: %.4f best k=%d" %
      (min_error,  best_n_neighbors))

min_error = 1
best_p = 0
for i in range(1, 11, 1):
    clf = KNeighborsClassifier(
        n_neighbors=best_n_neighbors, p=10**(i/10))
    clf.fit(train_data, train_target)
    test_accuracy = 1-clf.score(test_data, test_target)
    if test_accuracy < min_error:
        min_error = test_accuracy
        best_p = i/10
print("Test KNN min error rate: %.4f lg(p)= %.4f" %
      (min_error,  best_p))

min_error = 1
best_n_neighbors = 0
for i in range(1, 201, 5):
    clf = KNeighborsClassifier(n_neighbors=i, metric='chebyshev')
    clf.fit(train_data, train_target)
    test_accuracy = 1-clf.score(test_data, test_target)
    if test_accuracy < min_error:
        min_error = test_accuracy
        best_n_neighbors = i
print("Test KNN using Chebyshev Distance min error rate: %.4f best k=%d" %
      (min_error,  best_n_neighbors))


min_error = 1
best_n_neighbors = 0
for i in range(1, 201, 5):
    clf = KNeighborsClassifier(n_neighbors=i, metric='mahalanobis', algorithm='brute', metric_params={
        'V': np.cov(train_data)})
    clf.fit(train_data, train_target)
    test_accuracy = 1-clf.score(test_data, test_target)
    if test_accuracy < min_error:
        min_error = test_accuracy
        best_n_neighbors = i
print("Test KNN using Mahalanobis Distance min error rate: %.4f best k=%d" %
      (min_error,  best_n_neighbors))
