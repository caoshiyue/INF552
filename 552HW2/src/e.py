import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

data = pd.read_excel(".\\Data\\Folds5x2_pp.xlsx")
f = data.values
col = data.columns

# from (d)
multi_regression = [-1.97751311, -0.23391642,  0.06208294, -0.1580541]

# from (c)
simple_regression = [-2.171319958517794, -
                     1.1681351265557085, 1.4898716733991146, 0.45565010226298025]
plt.figure(figsize=(16, 9))
for i in range(0, 4):
    plt.plot(simple_regression[i], multi_regression[i],
             marker='o')
    plt.text(simple_regression[i], multi_regression[i], (round(simple_regression[i], 2),
                                                         round(multi_regression[i], 2)), ha='center', va='bottom', fontsize=10)

l = plt.legend(col, loc=2)
plt.gca().add_artist(l)
plt.xlabel("Simple regression")
plt.ylabel("Multiple regression")
plt.grid(True)
plt.show()
