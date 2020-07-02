import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.formula.api as sm

data = pd.read_excel(".\\Data\\Folds5x2_pp.xlsx")

col = data.columns
for i in range(0, 4):
    train_data = data[[col[i], col[-1]]]
    clf = sm.ols(formula=col[-1]+'~ ' + col[i] + '+ I('+col[i]+'**2) + I('+col[i]+'**3)',
                 data=train_data).fit()
    table = clf.summary()
    print(table)
