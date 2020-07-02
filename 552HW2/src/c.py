import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

data = pd.read_excel(".\\Data\\Folds5x2_pp.xlsx")
f = data.values
col = data.columns
train_target = f[:, -1]

# for (e)
simple_regression = []
q = 1
plt.figure(figsize=(16, 9))
for i in range(0, 4):
    plt.subplot(2, 2, q)
    plt.xlabel(col[i])
    plt.ylabel(col[4])
    plt.plot(f[:, i], f[:, 4], 'o', ms=1)
    q += 1
    train_data = f[:, i].reshape(-1, 1)
    clf = LinearRegression()
    clf.fit(train_data, train_target)
    x = [min(train_data), max(train_data)]
    y = clf.predict(x)
    plt.plot(x, y, 'r-')
    simple_regression.append(clf.coef_[0])

plt.show()
