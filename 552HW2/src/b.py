import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_excel(".\\Data\\Folds5x2_pp.xlsx")
f = data.values
print("Rows and columns :")
print(f.shape)
print("Each column represents:")
col = data.columns
print(col)


q = 1
plt.figure(figsize=(16, 9))
for i in range(0, 4):
    for j in range(i+1, 5):
        plt.subplot(4, 3, q)
        plt.xlabel(col[i])
        plt.ylabel(col[j])
        plt.plot(f[:, i], f[:, j], 'o', ms=1)
        q += 1
plt.tight_layout(pad=1, h_pad=0.1)
plt.show()

table = pd.DataFrame({
    "Mean": data.mean(),
    "Median": data.median(),
    "Range": (data.max()-data.min()),
    "First Quartiles": data.quantile(.25),
    "Third Quartiles": data.quantile(.75),
    "Interquartile Range": data.quantile(.75) - data.quantile(.25)
})
table
