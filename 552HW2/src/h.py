import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.formula.api as sm

data = pd.read_excel(".\\Data\\Folds5x2_pp.xlsx")
f = data.values
train_data = f[:, :-1]
train_target = f[:, -1]

col = data.columns
formula = col[-1]+' ~ '
for i in range(0, 4):
    formula += col[i]+'+'
for i in range(0, 4):
    for j in range(i, 4):
        formula += 'I(' + col[i]+'*'+col[j] + ')+'
formula = formula[:-1]

clf = sm.ols(formula=formula,
             data=data).fit()
table = clf.summary()
print(table)

formula = 'PE ~ V+AP+RH+I(AT*AT)+I(AT*V)+I(AT*RH)+I(AP*AP)+I(AP*RH)+I(RH*RH)'

shuffle = np.random.permutation(len(train_data))

test_size = int(len(f) * 0.3)
test_indexes = shuffle[:test_size]
train_indexes = shuffle[test_size:]

test_data = data.iloc[test_indexes]
train_data = data.iloc[train_indexes]

clf = sm.ols(formula=formula,
             data=train_data).fit()
pred = clf.predict(test_data.iloc[:, :-1])

pred_train = clf.predict(train_data.iloc[:, :4])
train_mse = sum((pred_train-train_data.PE).values**2) / \
    len(pred_train)-len(clf.params)

pred_test = clf.predict(test_data.iloc[:, :4])
test_mse = sum((pred_test-test_data.PE).values**2) / \
    len(pred_test)-len(clf.params)
print(train_mse)
print(test_mse)
