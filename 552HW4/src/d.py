import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("./src/feature.csv")
train_data = data.loc[data['train_test'] == 1]
train_data.loc[(train_data['activities']
                == 0) | (train_data['activities'] == 1), 'activities'] = 0
train_data.loc[(train_data['activities']
                != 0) & (train_data['activities'] != 1), 'activities'] = 1

features = ['max1', 'mean1', 'std1', 'max2',
            'mean2', 'std2', 'max6', 'mean6', 'std6', 'activities']
sns.pairplot(train_data[features], hue="activities", markers=["o", "x"])
plt.show()
