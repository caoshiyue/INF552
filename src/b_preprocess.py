import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


data = pd.read_csv(".\\data\\column_2C.dat", sep=' ', header=None)
f = data.values
f[:, 6][f[:, 6] == "AB"] = 1
f[:, 6][f[:, 6] == "NO"] = 0

class_0 = f[f[:, 6] == 0]
class_1 = f[f[:, 6] == 1]

q = 1
plt.figure(figsize=(16, 9))
for i in range(0, 5):
    for j in range(i+1, 6):
        plt.subplot(5, 3, q)
        plt.xlabel("Feature %d" % (i+1))
        plt.ylabel("Feature %d" % (j+1))
        plt.plot(class_0[:, i], class_0[:, j], 'rx', ms=3)
        plt.plot(class_1[:, i], class_1[:, j], 'go', ms=3)
        l = plt.legend(('Class 0', 'Class 1'), loc=2)
        plt.gca().add_artist(l)
        q += 1
plt.tight_layout(pad=1, h_pad=0.1)
plt.show()

q = 1
plt.figure(figsize=(16, 9))
for i in range(0, 6):
    plt.subplot(2, 3, q)
    plt.ylabel("Feature %d" % (i+1))
    plt.title("Box plots ")
    bp = plt.boxplot([class_0[:, i], class_1[:, i]],
                     whis=1.5, medianprops={'color': 'black'}, patch_artist=True, labels=['class0', 'class1'])
    colors = ['#4e72b8', '#f58220']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    q += 1
plt.tight_layout(pad=1, h_pad=1)
plt.show()

train_f = np.zeros([210, 7])
test_f = np.zeros([len(f)-210, 7])
train_f[0: 140, :] = class_1[0: 140, :]
train_f[140:, :] = class_0[0: 70, :]
test_f[0: 70, :] = class_1[140:, :]
test_f[70:, :] = class_0[70:, :]
clf = preprocessing.StandardScaler()
train_features = clf.fit_transform(train_f[:, 0: 6])
# use same transform mareix as training
test_features = clf.transform(test_f[:, 0: 6])

train_data = np.column_stack((train_features, train_f[:, 6]))
test_data = np.column_stack((test_features, test_f[:, 6]))
with open(".\\src\\train.csv", 'w', newline='') as t_file:
    csv_writer = csv.writer(t_file)
    for l in train_data:
        csv_writer.writerow(l)

with open(".\\src\\test.csv", 'w', newline='') as t_file:
    csv_writer = csv.writer(t_file)
    for l in test_data:
        csv_writer.writerow(l)
