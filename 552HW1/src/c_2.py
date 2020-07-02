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

train_accuracy = np.zeros([70, 1])
test_accuracy = np.zeros([70, 1])
min_error = 1
best_n_neighbors = 0
for i in range(208, -2, -3):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(train_data, train_target)
    train_accuracy[int((208-i)/3)] = 1-clf.score(train_data, train_target)
    test_accuracy[int((208-i)/3)] = 1-clf.score(test_data, test_target)
    if test_accuracy[int((208-i)/3)] < min_error:
        max_clf = clf
        best_n_neighbors = i
        min_error = test_accuracy[int((208-i)/3)]

k = np.linspace(208, 1, 70)
plt.figure(figsize=(16, 9))
plt.plot(k, train_accuracy, 'b--', marker='o')
plt.plot(k, test_accuracy, 'r--', marker='x')
plt.title("Error rate vs. K value")
plt.xticks(k)
l = plt.legend(('train accuracy', 'test accuracy'), loc=1)
plt.gca().add_artist(l)
ax = plt.gca()
ax.invert_xaxis()
ax.locator_params(nbins=35)  # sparse the axis ticks

print("Test KNN min error rate: %f n_neighbors= %f" %
      (min_error,  best_n_neighbors))
pred = max_clf.predict(test_data)
print('Confusion matrix')
print(confusion_matrix(test_target, pred))
tn, fp, fn, tp = confusion_matrix(test_target, pred).ravel()
print('True positive rate is %.2f' % (tp/tp+fn))
print('True negative rate is %.2f' % (tn/tn+fp))
print('True precision is %.2f' % (tp/(tp+fp)))
f1 = f1_score(test_target, pred, average='weighted')
print('F1-score %.2f' % f1)
plt.show()
