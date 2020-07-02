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
    clf = KNeighborsClassifier(
        n_neighbors=i, metric='euclidean', weights='distance')
    clf.fit(train_data, train_target)
    test_accuracy = 1-clf.score(test_data, test_target)
    if test_accuracy < min_error:
        min_error = test_accuracy
        best_n_neighbors = i
print("Test KNN using Euclidean Distance min error rate: %.4f best k=%d" %
      (min_error,  best_n_neighbors))

min_error = 1
best_n_neighbors = 0
for i in range(1, 201, 5):
    clf = KNeighborsClassifier(
        n_neighbors=i, metric='manhattan', weights='distance')
    clf.fit(train_data, train_target)
    test_accuracy = 1-clf.score(test_data, test_target)
    if test_accuracy < min_error:
        min_error = test_accuracy
        best_n_neighbors = i
print("Test KNN using Manhattan Distance min error rate: %.4f best k=%d" %
      (min_error,  best_n_neighbors))

min_error = 1
best_n_neighbors = 0
for i in range(1, 201, 5):
    clf = KNeighborsClassifier(
        n_neighbors=i, metric='chebyshev', weights='distance')
    clf.fit(train_data, train_target)
    test_accuracy = 1-clf.score(test_data, test_target)
    if test_accuracy < min_error:
        min_error = test_accuracy
        best_n_neighbors = i
print("Test KNN using Chebyshev Distance min error rate: %.4f best k=%d" %
      (min_error,  best_n_neighbors))
