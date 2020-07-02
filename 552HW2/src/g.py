import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.formula.api as sm

data = pd.read_excel(".\\Data\\Folds5x2_pp.xlsx")

col = data.columns
formula = col[-1]+' ~ '
for i in range(0, 4):
    formula += col[i]+'+'
for i in range(0, 3):
    for j in range(i+1, 4):
        formula += 'I(' + col[i]+'*'+col[j] + ')+'
formula = formula[:-1]

clf = sm.ols(formula=formula,
             data=data).fit()
table = clf.summary()
print(table)
