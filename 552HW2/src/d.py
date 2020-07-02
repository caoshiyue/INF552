import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm

data = pd.read_excel(".\\Data\\Folds5x2_pp.xlsx")
f = data.values
col = data.columns
train_data = f[:, :-1]
train_target = f[:, -1]


X2 = sm.add_constant(train_data)
est = sm.OLS(train_target, X2)
clf = est.fit()
table = clf.summary()
print(table)

# for next question
multi_regression = clf.params[1:5]
