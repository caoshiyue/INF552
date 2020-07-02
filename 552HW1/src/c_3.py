from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

f = np.loadtxt(
    open(".\\src\\train.csv", "rb"), delimiter=",", skiprows=0)
train_data = f[:, :-1]
train_target = f[:, -1]
f = np.loadtxt(
    open(".\\src\\test.csv", "rb"), delimiter=",", skiprows=0)
test_data = f[:, :-1]
test_target = f[:, -1]

min_error_record = []
for N in range(10, 220, 10):
    sub_train_data = np.append(train_data[0:round(N-(N/3))],
                               train_data[140:(140+round(N/3))], axis=0)

    sub_train_target = np.append(train_target[0:round(N-(N/3))],
                                 train_target[140:(140+round(N/3))], axis=0)
    min_error = 1
    for i in range(1, N, 5):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(sub_train_data, sub_train_target)
        test_accuracy = 1-clf.score(test_data, test_target)
        if test_accuracy < min_error:
            min_error = test_accuracy
    min_error_record.append(min_error)

k = range(10, 220, 10)
plt.figure(figsize=(16, 9))
plt.plot(k, min_error_record, '--', marker='o')
plt.xticks(k)
plt.title("Learning Curve")
plt.xlabel("N")
plt.ylabel("Error rate")
plt.show()
