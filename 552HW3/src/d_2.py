import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from break_data import brk_data
from extract_feature import ext_feature

brk_data(2)
path = "./breaking_data"
feature = ext_feature(path, 2, 0)
train_data = feature.loc[feature['train_test'] == 1]
train_data.loc[(train_data['activities']
                == 0) | (train_data['activities'] == 1), 'activities'] = 0
train_data.loc[(train_data['activities']
                != 0) & (train_data['activities'] != 1), 'activities'] = 1

features = ['max1', 'mean1', 'std1', 'max2',
            'mean2', 'std2', 'max12', 'mean12', 'std12', 'activities']
sns.pairplot(train_data[features], hue="activities", markers=["o", "x"])
plt.show()
